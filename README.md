# Desktop Rotaeno Stabilizer

![a screenshot of the application in question](./asset/example.png)

# [See this in action!](https://youtu.be/qs2fVbXWDNI)

## About

Rotaeno is uniquely disadvantaged in that a screen recording on its own is straight up unusable. Currently, players either try a handcam, or use existing stabilization tools. I want everyone to pass around cool looking rotaeno gameplay and I think the best way to achieve this is by making it as frictionless as possible to produce something good looking.

Thus, this stabilizer only requires a screen recording with [Stream Encoding V2](https://rotaeno.com/streaming) enabled. The rest is efficiently handled for you. 

## Features

- realtime playback!!!!! (I've tested up to 120fps)
- fast video encoding 
- auto crop for tablets 
- pretty ui (blurred background, ring, device border)
- customization

## Download

The most recent successful CI run in the [Actions](<https://github.com/paoda/rota-stabilizer/actions>) tab will have binary artifacts uploaded. 

Alternatively: 

[Windows (nightly.link)](<https://nightly.link/paoda/rota-stabilizer/workflows/main/main/rota-stabilizer-windows-latest.zip.zip>)

[macOS (nightly.link)](<https://nightly.link/paoda/rota-stabilizer/workflows/main/main/rota-stabilizer-macos-latest.zip.zip>)


## How to Use

- TODO: Instructions on how to use the GUI

**NOTE:** This tool uses [ffmpeg](https://www.ffmpeg.org/), so it can work with many video file types. Feel free to have an `input.mkv`, `output.webm`, or `input.mov`. 

# Building

`zig build -Doptimize=ReleaseSafe`

The binary can then be found in `zig-out/bin`. 

# Related Work
- [Lawrenceeeeeeee/python_rotaeno_stabilizer](https://github.com/Lawrenceeeeeeee/python_rotaeno_stabilizer)
- [linnaea/rotaeno-stablizer](https://github.com/linnaea/rotaeno-stablizer)
- [I-love-study/py-rotaeno-stablizer-gui](https://github.com/I-love-study/py-rotaeno-stablizer-gui)
- [chinosk6/rotaeno-stabilizer-front](https://github.com/chinosk6/rotaeno-stabilizer-front)
