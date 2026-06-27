# Desktop Rotaeno Stabilizer

![a screenshot of the application in question](./asset/example.png)

## [See this in action!](https://youtu.be/raAOzAHmAFM)

## About

Rotaeno is uniquely disadvantaged in that a screen recording on its own is straight up unusable. Currently, players either try a handcam, or use existing stabilization tools. I want everyone to pass around cool looking rotaeno gameplay and I think the best way to achieve this is by making it as frictionless as possible to produce something good looking.

Thus, this stabilizer only requires a screen recording with [Streaming Mode](https://rotaeno.com/streaming) enabled. The rest is efficiently handled for you.

## Features

- realtime playback!!!!! (I've tested up to 120fps)
- fast video encoding
- auto crop for tablets
- pretty ui (blurred background, ring, device border)
- ui customization
- easy upload from mobile device via QR Code
- transparent background (for OBS) support

## Download (Nightly)

The most recent successful CI run in the [Actions](<https://github.com/paoda/rota-stabilizer/actions>) tab will have binary artifacts uploaded (Linux users can find the flatpak here).

Alternatively:

[Windows (nightly.link)](<https://nightly.link/paoda/rota-stabilizer/workflows/main/main/rota-stabilizer-windows-latest.zip.zip>)

[macOS (nightly.link)](<https://nightly.link/paoda/rota-stabilizer/workflows/main/main/rota-stabilizer-macos-latest.zip.zip>)

## How to Use

1. Click on the Upload Tab to find the QR Code that allows you to upload a screen recording from your mobile device
    - Take note of the tooltip above the QR Code which informs you where your uploads will be saved

2. Scan the QR Code, then, on your mobile device, click 'Browse...' and then click Upload
3. To the right of the "Input Video Path" input field, click 'Browse...' and select the uploaded file
4. Optionally, choose an output path.
5. Feel free to use the Render and Hardware & Output tabs to configure the stabilizer to your liking.
6. For Realtime, playback, hit Play. To produce a video file, hit Encode
7. The stabilized footage will be saved to your chosen output path. If you didn't select one, it will default to `rota-stabilizer/` in your `Videos/` (Win/Linux), or `Movies` (macOS) folder.

#### How to Enable Transparency

In rota-stabilizer:
1. Disable the background in the "Render" tab

In OBS (Windows):
1. Add a "Game Capture" source to the current scene
2. Select "Capture specific window" from the "Mode" dropdown
3. In the "Window" dropdown, choose the option that says "Rotaeno Stabilizer" (rota-stabilizer must already be running)
4. Check "Allow Transparency", then click OK

## Building

### macOS

```sh
brew install ffmpeg
zig build -Doptimize=ReleaseSafe
```

### Windows

```sh
zig build -Doptimize=ReleaseSafe
```

### Linux

- TODO: i feel like this really depends on the desktop environment

The binary and its dependencies can then be found in `zig-out/bin`.

## Related Work

- [Lawrenceeeeeeee/python_rotaeno_stabilizer](https://github.com/Lawrenceeeeeeee/python_rotaeno_stabilizer)
- [linnaea/rotaeno-stablizer](https://github.com/linnaea/rotaeno-stablizer)
- [I-love-study/py-rotaeno-stablizer-gui](https://github.com/I-love-study/py-rotaeno-stablizer-gui)
- [chinosk6/rotaeno-stabilizer-front](https://github.com/chinosk6/rotaeno-stabilizer-front)
