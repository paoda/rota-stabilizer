const std = @import("std");

fn buildCheck(b: *std.Build, mod: *std.Build.Module) !void {
    const step = b.step("check", "check if rota-stabilizer compiles");

    const exe = b.addExecutable(.{
        .name = "rota-stabilizer",
        .root_module = mod,
    });

    step.dependOn(&exe.step);
}

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // We will also create a module for our other entry point, 'main.zig'.
    const exe_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    exe_mod.addAnonymousImport("build.zig.zon", .{ .root_source_file = b.path("build.zig.zon") });
    exe_mod.addAnonymousImport("asset/Inter-Medium.ttf", .{ .root_source_file = b.path("asset/Inter-Medium.ttf") });

    const qrcodegen = b.dependency("zqrcodegen", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("qrcodegen", qrcodegen.module("zqrcodegen"));

    const znfde = b.dependency("znfde", .{ .target = target, .optimize = optimize, .with_portal = true });
    exe_mod.addImport("znfde", znfde.module("root"));
    exe_mod.linkLibrary(znfde.artifact("nfde"));

    const known_folders = b.dependency("known_folders", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("known-folders", known_folders.module("known-folders"));

    const sdl = switch (target.result.os.tag) {
        .macos => blk: {
            const result = try std.process.Child.run(.{
                .allocator = b.allocator,
                .argv = &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            });

            const sdk_root = std.mem.trim(u8, result.stdout, " \n");

            break :blk b.dependency("sdl", .{
                .target = target,
                .optimize = .ReleaseFast,
                .preferred_linkage = .static,
                .system_include_path = b.pathJoin(&.{ sdk_root, "usr", "include" }),
                .system_framework_path = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }),
                .library_path = b.pathJoin(&.{ sdk_root, "usr", "lib" }),
            });
        },
        else => b.dependency("sdl", .{ .target = target, .optimize = .ReleaseFast, .preferred_linkage = .static }),
    };

    const sdl_lib = sdl.artifact("SDL3");
    exe_mod.linkLibrary(sdl_lib);

    const zgui = b.dependency("zgui", .{ .target = target, .optimize = optimize, .shared = false, .backend = .sdl3_opengl3 });
    exe_mod.addImport("zgui", zgui.module("root"));

    const zgui_lib = zgui.artifact("imgui");
    zgui_lib.linkLibrary(sdl_lib);
    exe_mod.linkLibrary(zgui_lib);

    switch (target.result.os.tag) {
        .windows => {
            const ffmpeg = b.lazyDependency("ffmpeg", .{}) orelse return;
            exe_mod.addIncludePath(ffmpeg.path("include/"));
            exe_mod.addLibraryPath(ffmpeg.path("lib/"));

            const ffmpeg_libs = [_][]const u8{ "avcodec", "avformat", "avfilter", "swscale", "avutil", "swresample" };

            const base_path = ffmpeg.path("bin" ++ std.fs.path.sep_str);
            const dir = try base_path.getPath3(b, null).openDir(".", .{ .iterate = true });

            var walk = try dir.walk(b.allocator);
            defer walk.deinit();

            while (try walk.next()) |entry| {
                const lib = containsAny(ffmpeg_libs[0..], entry.basename) orelse continue;
                const src_path = try base_path.join(b.allocator, entry.basename);

                // b.installBinFile doesn't support LazyPath for some reason :\
                b.getInstallStep().dependOn(&b.addInstallFileWithDir(src_path, .bin, entry.basename).step);
                exe_mod.linkSystemLibrary(lib, .{});
            }
        },
        else => {
            exe_mod.linkSystemLibrary("avcodec", .{});
            exe_mod.linkSystemLibrary("avformat", .{});
            exe_mod.linkSystemLibrary("avfilter", .{});
            exe_mod.linkSystemLibrary("swscale", .{});
            exe_mod.linkSystemLibrary("swresample", .{});
            exe_mod.linkSystemLibrary("avutil", .{});
        },
    }

    const gl_mod = @import("zigglgen").generateBindingsModule(b, .{
        .api = .gl,
        .version = .@"3.3",
        .profile = .core,
    });
    exe_mod.addImport("gl", gl_mod);

    const enable_tracy = b.option(bool, "tracy", "Enable Tracy Profiling") orelse false;
    const tracy = b.dependency("tracy", .{ .target = target, .optimize = optimize });
    const tracy_impl = if (enable_tracy) "tracy_impl_enabled" else "tracy_impl_disabled";

    exe_mod.addImport("tracy", tracy.module("tracy"));
    exe_mod.addImport("tracy_impl", tracy.module(tracy_impl));

    try buildCheck(b, exe_mod);

    // This creates another `std.Build.Step.Compile`, but this one builds an executable
    // rather than a static library.
    const exe = b.addExecutable(.{
        .name = "rota-stabilizer",
        .root_module = exe_mod,
        .use_llvm = true,
    });

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}

/// will return the first match
fn containsAny(needles: []const []const u8, haystack: []const u8) ?[]const u8 {
    for (needles) |needle| {
        if (std.mem.containsAtLeast(u8, haystack, 1, needle)) return needle;
    }

    return null;
}
