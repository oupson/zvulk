const std = @import("std");
const builtin = @import("builtin");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    const output_frag, const output_vertex = build_shaders: {
        switch (builtin.target.os.tag) {
            .linux => {
                const tool = b.addExecutable(.{
                    .name = "shader_compile",
                    .root_source_file = b.path("tools/shader_compile.zig"),
                    .target = b.graph.host,
                });

                tool.linkSystemLibrary("shaderc");
                tool.linkLibC();

                const tool_step_fragment = b.addRunArtifact(tool);
                tool_step_fragment.addFileArg(b.path("shaders/shader.frag"));
                const output_frag = tool_step_fragment.addOutputFileArg("fragment.spv");

                const tool_step_vertex = b.addRunArtifact(tool);
                tool_step_vertex.addFileArg(b.path("shaders/shader.vert"));
                const output_vertex = tool_step_vertex.addOutputFileArg("vertex.spv");
                break :build_shaders .{ output_frag, output_vertex };
            },
            .windows => {
                const a = [_][]const u8{
                    "glslc.exe",
                };
                const tool_step_fragment = b.addSystemCommand(&a);
                tool_step_fragment.addFileArg(b.path("shaders/shader.frag"));
                const output_frag = tool_step_fragment.addPrefixedOutputFileArg("-o", "fragment.spv");

                const tool_step_vertex = b.addSystemCommand(&a);
                tool_step_vertex.addFileArg(b.path("shaders/shader.vert"));
                const output_vertex = tool_step_vertex.addPrefixedOutputFileArg("-o", "vertex.spv");

                break :build_shaders .{ output_frag, output_vertex };
            },
            else => {
                unreachable;
            },
        }
    };

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "zvulk",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "zvulk",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    switch (builtin.target.os.tag) {
        .linux => {
            const Scanner = @import("wayland").Scanner;

            const scanner = Scanner.create(b, .{});

            const wayland = b.createModule(.{ .root_source_file = scanner.result });

            scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");
            scanner.addSystemProtocol("unstable/xdg-decoration/xdg-decoration-unstable-v1.xml");
            scanner.addSystemProtocol("unstable/pointer-constraints/pointer-constraints-unstable-v1.xml");
            scanner.addSystemProtocol("unstable/relative-pointer/relative-pointer-unstable-v1.xml");

            scanner.generate("wl_compositor", 1);
            scanner.generate("wl_shm", 1);
            scanner.generate("xdg_wm_base", 6);
            scanner.generate("wl_output", 4);
            scanner.generate("zxdg_decoration_manager_v1", 1);
            scanner.generate("wl_seat", 8);
            scanner.generate("zwp_pointer_constraints_v1", 1);
            scanner.generate("zwp_relative_pointer_manager_v1", 1);

            exe.root_module.addImport("wayland", wayland);

            exe.linkSystemLibrary("wayland-client");
            exe.linkSystemLibrary("xkbcommon");

            exe.linkSystemLibrary("vulkan");
        },
        .windows => {
            var env = try std.process.getEnvMap(b.allocator);
            defer env.deinit();

            // TODO: arch
            if (env.get("VCPKG_ROOT")) |root| {
                const lib_path = try std.fs.path.join(b.allocator, &.{ root, "installed", "x64-windows", "lib" });
                defer b.allocator.free(lib_path);
                const include_path = try std.fs.path.join(b.allocator, &.{ root, "installed", "x64-windows", "include" });
                defer b.allocator.free(lib_path);
                exe.addLibraryPath(.{ .cwd_relative = lib_path });
                exe.addIncludePath(.{ .cwd_relative = include_path });
            }

            const static_libs = [_][]const u8{
                "user32",
            };
            for (static_libs) |libl| exe.linkSystemLibrary(libl);

            exe.linkSystemLibrary("vulkan-1");
        },
        else => {
            unreachable;
        },
    }

    exe.linkLibC();

    exe.root_module.addAnonymousImport("shaders/fragment.spv", .{
        .root_source_file = output_frag,
    });

    exe.root_module.addAnonymousImport("shaders/vertex.spv", .{
        .root_source_file = output_vertex,
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

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
