const std = @import("std");
const zigglgen = @import("zigglgen");

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
    });

    exe_mod.addAnonymousImport("build.zig.zon", .{ .root_source_file = b.path("build.zig.zon") });
    exe_mod.addAnonymousImport("asset/Inter-Medium.ttf", .{ .root_source_file = b.path("asset/Inter-Medium.ttf") });

    const gl_mod = zigglgen.generateBindingsModule(b, .{ .api = .gl, .version = .@"3.3", .profile = .core });
    exe_mod.addImport("gl", gl_mod);

    const qrcodegen = b.dependency("zqrcodegen", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("qrcodegen", qrcodegen.module("zqrcodegen"));

    const znfde = b.dependency("znfde", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("znfde", znfde.module("root"));
    exe_mod.linkLibrary(znfde.artifact("nfde"));

    const known_folders = b.dependency("known_folders", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("known-folders", known_folders.module("known-folders"));

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const sdl = b.dependency("sdl", .{ .target = target, .optimize = optimize });
    translate_c.addIncludePath(sdl.path("include"));

    const zimgui = b.dependency("zimgui", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("zimgui", zimgui.module("dcimgui"));

    // -- ffmpeg --
    const libs = [_][]const u8{ "avcodec", "avformat", "avfilter", "swscale", "avutil", "swresample" };

    const ffmpeg_dep: ?*std.Build.Dependency = switch (target.result.os.tag) {
        .windows => blk: {
            const dep = b.lazyDependency("ffmpeg", .{}) orelse return;
            translate_c.addIncludePath(dep.path("include/"));

            break :blk dep;
        },
        else => blk: {
            for (libs) |lib| translate_c.linkSystemLibrary(lib, .{});
            break :blk null;
        },
    };

    const c_mod = translate_c.createModule();
    exe_mod.addImport("c", c_mod);
    c_mod.linkLibrary(sdl.artifact("SDL3"));

    if (ffmpeg_dep) |dep| {
        c_mod.addLibraryPath(dep.path("lib/"));

        const base_lazy_path = dep.path("bin" ++ std.fs.path.sep_str);
        const base_dir = blk: {
            const path = try base_lazy_path.getPath4(b, null);
            break :blk try path.openDir(b.graph.io, ".", .{ .iterate = true });
        };

        var walk = try base_dir.walk(b.allocator);
        defer walk.deinit();

        while (try walk.next(b.graph.io)) |entry| {
            const lib = containsAny(libs[0..], entry.basename) orelse continue;
            const src_path = try base_lazy_path.join(b.allocator, entry.basename);

            // b.installBinFile doesn't support LazyPath for some reason :\
            b.getInstallStep().dependOn(&b.addInstallFileWithDir(src_path, .bin, entry.basename).step);
            c_mod.linkSystemLibrary(lib, .{});
        }
    }

    const enable_tracy = b.option(bool, "tracy", "Enable Tracy Profiling") orelse false;

    const tracy = b.dependency("tracy", .{ .target = target, .optimize = optimize });
    exe_mod.addImport("tracy", tracy.module("tracy"));
    exe_mod.addImport("tracy_impl", tracy.module(if (enable_tracy) "tracy_impl_enabled" else "tracy_impl_disabled"));

    try check(b, exe_mod);

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

fn check(b: *std.Build, mod: *std.Build.Module) !void {
    const step = b.step("check", "check if rota-stabilizer compiles");

    const exe = b.addExecutable(.{ .name = "rota-stabilizer", .root_module = mod });
    step.dependOn(&exe.step);
}

/// will return the first match
fn containsAny(needles: []const []const u8, haystack: []const u8) ?[]const u8 {
    for (needles) |needle| {
        if (std.mem.containsAtLeast(u8, haystack, 1, needle)) return needle;
    }

    return null;
}
