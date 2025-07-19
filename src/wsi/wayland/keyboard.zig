const std = @import("std");

pub const xkbcommon = @cImport({
    @cInclude("xkbcommon/xkbcommon.h");
});

const Self = @This();

context: *xkbcommon.xkb_context,
keymap: ?*xkbcommon.xkb_keymap = null,
state: ?*xkbcommon.xkb_state = null,

pub fn init() !Self {
    // todo: error
    return Self{
        .context = xkbcommon.xkb_context_new(xkbcommon.XKB_CONTEXT_NO_FLAGS) orelse return error.FailedToCreateContext,
    };
}

pub fn deinit(self: *Self) void {
    xkbcommon.xkb_keymap_unref(self.keymap);
    xkbcommon.xkb_state_unref(self.state);
    xkbcommon.xkb_context_unref(self.context);
}

pub fn parseKeymap(self: *Self, fd: i32, size: u32) !void {
    const map_shm = try std.posix.mmap(null, @intCast(size), std.posix.PROT.READ, .{ .TYPE = .SHARED }, fd, 0);
    defer {
        std.posix.munmap(map_shm);
        std.posix.close(fd);
    }

    const xkb_keymap = xkbcommon.xkb_keymap_new_from_buffer(
        self.context,
        map_shm.ptr,
        map_shm.len,
        xkbcommon.XKB_KEYMAP_FORMAT_TEXT_V1,
        xkbcommon.XKB_KEYMAP_COMPILE_NO_FLAGS,
    ) orelse return error.FailedToParseKeymap;

    const xkb_state = xkbcommon.xkb_state_new(xkb_keymap) orelse return error.FailedToCreateState;

    xkbcommon.xkb_keymap_unref(self.keymap);
    xkbcommon.xkb_state_unref(self.state);
    self.keymap = xkb_keymap;
    self.state = xkb_state;
}

pub fn setModifier(self: *Self, mods_depressed: u32, mods_latched: u32, mods_locked: u32, group: u32) void {
    _ = xkbcommon.xkb_state_update_mask(self.state, mods_depressed, mods_latched, mods_locked, 0, 0, group);
}

pub fn parseKeyCode(self: *Self, key: u32) u32 {
    const keycode = key + 8;
    const sym = xkbcommon.xkb_state_key_get_one_sym(self.state, keycode);
    return sym;
}

pub fn toLower(key: u32) u32 {
    return xkbcommon.xkb_keysym_to_lower(key);
}
