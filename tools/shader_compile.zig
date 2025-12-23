const std = @import("std");
const shaderc = @cImport({
    @cInclude("shaderc/shaderc.h");
});

const usage =
    \\Usage: ./shader_compile INPUT OUTPUT
    \\
;

const ShaderType = enum {
    fragment,
    vertex,
    unknown,

    fn toShaderc(self: @This()) c_uint {
        return switch (self) {
            .fragment => shaderc.shaderc_glsl_fragment_shader,
            .vertex => shaderc.shaderc_glsl_vertex_shader,
            .unknown => shaderc.shaderc_glsl_infer_from_source,
        };
    }
};

pub fn main() !void {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const args = try std.process.argsAlloc(arena);
    defer std.process.argsFree(arena, args);

    if (args.len != 3) {
        const stderr = std.Progress.lockStderrWriter(&.{});
        defer std.Progress.unlockStderrWriter();
        defer stderr.flush() catch {};
        try stderr.writeAll(usage);
        return error.MissingInputOutput;
    }

    const input_file_path = args[1];
    const output_file_path = args[2];

    const extension = std.fs.path.extension(input_file_path);
    var shader_type = ShaderType.unknown;

    if (std.mem.eql(u8, extension, ".frag")) {
        shader_type = .fragment;
    } else if (std.mem.eql(u8, extension, ".vert")) {
        shader_type = .vertex;
    } else {
        std.log.warn("unknown shader type: \"{s}\", trying to guess ...", .{extension});
    }

    const compiler = shaderc.shaderc_compiler_initialize() orelse return error.FailedToInitCompiler;
    defer shaderc.shaderc_compiler_release(compiler);

    const cwd = std.fs.cwd();
    const input_file = try cwd.openFile(input_file_path, .{ .mode = .read_only });
    defer input_file.close();
    const meta = try input_file.stat();

    const input_buffer = try arena.alloc(u8, meta.size);
    defer arena.free(input_buffer);
    _ = try input_file.readAll(input_buffer);

    const debug_input_path = try arena.allocSentinel(u8, input_file_path.len, 0);
    defer arena.free(debug_input_path);
    @memcpy(debug_input_path, input_file_path);

    const options = shaderc.shaderc_compile_options_initialize();
    defer shaderc.shaderc_compile_options_release(options);

    shaderc.shaderc_compile_options_set_optimization_level(options, shaderc.shaderc_optimization_level_performance);

    const result = shaderc.shaderc_compile_into_spv(
        compiler,
        input_buffer.ptr,
        input_buffer.len,
        shader_type.toShaderc(),
        debug_input_path,
        "main",
        options,
    );
    defer shaderc.shaderc_result_release(result);

    const status = shaderc.shaderc_result_get_compilation_status(result);
    if (status != shaderc.shaderc_compilation_status_success) {
        const error_message: [*:0]const u8 = shaderc.shaderc_result_get_error_message(result);
        const stderr = std.Progress.lockStderrWriter(&.{});
        defer std.Progress.unlockStderrWriter();
        defer stderr.flush() catch {};
        try stderr.writeAll(std.mem.span(error_message));
        return error.CompilationFailed;
    }

    const bytes = shaderc.shaderc_result_get_bytes(result);
    const bytes_len = shaderc.shaderc_result_get_length(result);
    const output_file = try cwd.createFile(output_file_path, .{});
    defer output_file.close();

    try output_file.writeAll(bytes[0..bytes_len]);
}
