"""Minimal GLFW MuJoCo viewer with policy-camera and prompt overlays.

The stock passive viewer exposes ``Handle.set_images()``, but on the Windows
MuJoCo 3.11 build those images are not presented reliably.  This viewer owns
the OpenGL context and draws the main scene and both RGB observations into the
same framebuffer, so their render order is explicit.
"""

from __future__ import annotations

from collections.abc import Sequence

import glfw
import mujoco
import numpy as np


def _ascii_text(text: str) -> str:
    """Return text supported by MuJoCo's built-in 7-bit bitmap font."""
    return "".join(character if 32 <= ord(character) < 127 else "?" for character in text)


def _fit_text(text: str, widths: Sequence[int], maximum_width: int) -> str:
    """Truncate one line to a pixel width, preserving a visible ellipsis."""
    if maximum_width <= 0:
        return ""
    text = _ascii_text(text.replace("\n", " "))

    def width(value: str) -> int:
        return sum(int(widths[ord(character)]) for character in value)

    if width(text) <= maximum_width:
        return text
    ellipsis = "..."
    available = maximum_width - width(ellipsis)
    if available <= 0:
        return ""
    used = 0
    fitted: list[str] = []
    for character in text:
        character_width = int(widths[ord(character)])
        if used + character_width > available:
            break
        fitted.append(character)
        used += character_width
    return "".join(fitted).rstrip() + ellipsis


def _wrap_text(
    text: str,
    widths: Sequence[int],
    maximum_width: int,
    maximum_lines: int = 2,
) -> list[str]:
    """Wrap text on spaces and truncate only when all allowed lines are full."""
    remaining = " ".join(_ascii_text(text).split())
    lines: list[str] = []
    while remaining and len(lines) < maximum_lines:
        if len(lines) == maximum_lines - 1:
            lines.append(_fit_text(remaining, widths, maximum_width))
            break

        used = 0
        last_space = -1
        cut = len(remaining)
        for index, character in enumerate(remaining):
            used += int(widths[ord(character)])
            if character == " ":
                last_space = index
            if used > maximum_width:
                cut = last_space if last_space > 0 else index
                break
        line = remaining[:cut].rstrip()
        if not line:
            line = _fit_text(remaining, widths, maximum_width)
            remaining = ""
        else:
            remaining = remaining[cut:].lstrip()
        lines.append(line)
    return lines


def overlay_rects(
    viewport_width: int,
    viewport_height: int,
    image_aspect: float,
) -> tuple[mujoco.MjrRect, mujoco.MjrRect] | None:
    """Return equally sized top-left and top-right overlay rectangles."""
    if viewport_width <= 0 or viewport_height <= 0 or image_aspect <= 0:
        return None
    margin = max(8, round(viewport_width * 0.008))
    available_width = max(1, viewport_width - 3 * margin)
    panel_width = min(round(viewport_width * 0.24), available_width // 2)
    if panel_width < 32:
        return None
    panel_height = min(round(panel_width * image_aspect), viewport_height - 2 * margin)
    if panel_height < 32:
        return None
    panel_width = max(32, round(panel_height / image_aspect))
    bottom = viewport_height - margin - panel_height
    return (
        mujoco.MjrRect(margin, bottom, panel_width, panel_height),
        mujoco.MjrRect(
            viewport_width - margin - panel_width,
            bottom,
            panel_width,
            panel_height,
        ),
    )


def overlay_rects_many(
    viewport_width: int,
    viewport_height: int,
    image_aspect: float,
    count: int,
) -> tuple[mujoco.MjrRect, ...]:
    """Return a compact top-row layout for one or more camera thumbnails."""
    if count <= 0 or viewport_width <= 0 or viewport_height <= 0 or image_aspect <= 0:
        return ()
    margin = max(8, round(viewport_width * 0.008))
    available_width = max(1, viewport_width - (count + 1) * margin)
    panel_width = min(round(viewport_width * 0.20), available_width // count)
    panel_height = min(round(panel_width * image_aspect), viewport_height - 2 * margin)
    if panel_width < 32 or panel_height < 32:
        return ()
    panel_width = max(32, round(panel_height / image_aspect))
    bottom = viewport_height - margin - panel_height
    return tuple(
        mujoco.MjrRect(margin + index * (panel_width + margin), bottom, panel_width, panel_height)
        for index in range(count)
    )


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize RGB uint8 data with dependency-free nearest-neighbour sampling."""
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    y = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
    x = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
    return np.ascontiguousarray(image[y[:, None], x[None, :]])


class GLFWViewer:
    """A panel-free interactive viewer rendered entirely through MuJoCo/GLFW."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        title: str = "MuJoCo: SO101 tabletop",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        self._window = glfw.create_window(width, height, title, None, None)
        if self._window is None:
            raise RuntimeError("GLFW could not create the MuJoCo viewer window")

        self.model = model
        self.data = data
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10_000)
        self.context: mujoco.MjrContext | None = None
        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_cursor = (0.0, 0.0)

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self._install_callbacks()

    def _install_callbacks(self) -> None:
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_scroll_callback(self._window, self._on_scroll)

    def _on_key(self, window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def _on_mouse_button(self, window, _button: int, _action: int, _mods: int) -> None:
        self._button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self._button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos: float, ypos: float) -> None:
        if not (self._button_left or self._button_middle or self._button_right):
            return
        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        width, height = glfw.get_window_size(window)
        if height <= 0:
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model,
            action,
            (xpos - last_x) / max(width, 1),
            (ypos - last_y) / height,
            self.camera,
        )

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.camera,
        )

    @property
    def viewport(self) -> mujoco.MjrRect | None:
        if not self.is_running():
            return None
        width, height = glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            return None
        return mujoco.MjrRect(0, 0, width, height)

    def is_running(self) -> bool:
        return self._window is not None and not glfw.window_should_close(self._window)

    def render(
        self,
        images: Sequence[np.ndarray] = (),
        *,
        prompt: str | None = None,
    ) -> None:
        if not self.is_running() or self.context is None:
            return
        glfw.make_context_current(self._window)
        viewport = self.viewport
        if viewport is None:
            # Minimized Windows framebuffers can be 0x0. Keep processing
            # restore/close events without drawing into an invalid viewport.
            glfw.poll_events()
            return
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if len(images) >= 2:
            self._draw_observations(viewport, images)
        if prompt:
            self._draw_prompt(viewport, prompt)
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def _draw_observations(
        self,
        viewport: mujoco.MjrRect,
        images: Sequence[np.ndarray],
    ) -> None:
        assert self.context is not None
        aspect = images[0].shape[0] / images[0].shape[1]
        rects = overlay_rects_many(viewport.width, viewport.height, aspect, len(images))
        if not rects:
            return
        labels = ("global", "wrist")
        for index, (rect, image) in enumerate(zip(rects, images, strict=False)):
            border = 3
            mujoco.mjr_rectangle(
                mujoco.MjrRect(
                    rect.left - border,
                    rect.bottom - border,
                    rect.width + 2 * border,
                    rect.height + 2 * border,
                ),
                0.05,
                0.05,
                0.05,
                1.0,
            )
            resized = resize_rgb(image, rect.height, rect.width)
            # MuJoCo camera arrays use a top-left origin while OpenGL pixels use
            # a bottom-left origin.
            pixels = np.ascontiguousarray(np.flip(resized, axis=0)).reshape(-1)
            mujoco.mjr_drawPixels(pixels, None, rect, self.context)
            # Keep the three views identifiable while inspecting a trajectory.
            label = labels[index] if index < len(labels) else f"camera {index + 1}"
            mujoco.mjr_text(
                # MuJoCo's Python enum does not expose a SMALL font on all
                # builds (including the Windows wheels), so use the portable
                # normal bitmap font for camera labels.
                mujoco.mjtFont.mjFONT_NORMAL.value,
                label,
                self.context,
                rect.left / max(viewport.width, 1),
                (rect.bottom + rect.height - 18) / max(viewport.height, 1),
                1.0,
                1.0,
                1.0,
            )

    def _draw_prompt(self, viewport: mujoco.MjrRect, prompt: str) -> None:
        """Draw the current policy prompt centred along the bottom edge."""
        if viewport.width <= 0 or viewport.height <= 0:
            return
        assert self.context is not None
        font = mujoco.mjtFont.mjFONT_NORMAL.value
        widths = self.context.charWidth
        char_height = int(self.context.charHeight)
        horizontal_padding = max(16, round(viewport.width * 0.02))
        lines = _wrap_text(
            f"Prompt: {prompt}",
            widths,
            viewport.width - 2 * horizontal_padding,
        )
        if not lines:
            return

        line_gap = max(4, char_height // 4)
        content_height = len(lines) * char_height + (len(lines) - 1) * line_gap
        band_height = max(42, content_height + 18)
        mujoco.mjr_rectangle(
            mujoco.MjrRect(0, 0, viewport.width, band_height),
            0.03,
            0.03,
            0.03,
            0.82,
        )
        bottom_padding = max(0, (band_height - content_height) // 2)
        for index, line in enumerate(lines):
            text_width = sum(int(widths[ord(character)]) for character in line)
            x = max(0.0, (viewport.width - text_width) / (2.0 * viewport.width))
            line_from_bottom = len(lines) - index - 1
            y_pixels = bottom_padding + line_from_bottom * (char_height + line_gap)
            y = y_pixels / max(viewport.height, 1)
            mujoco.mjr_text(font, line, self.context, x, y, 0.96, 0.96, 0.96)

    def close(self) -> None:
        if self.context is not None:
            glfw.make_context_current(self._window)
            self.context.free()
            self.context = None
        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None


__all__ = ["GLFWViewer", "overlay_rects", "overlay_rects_many", "resize_rgb"]
