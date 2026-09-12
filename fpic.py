#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fpic.py —— 文件 ⇄ 图片互转工具（PNG，无损）

用法:
    python3 fpic.py encode 输入文件 输出.png [--width W] [--base B]
    python3 fpic.py decode 输入.png [--out-dir 目录]

像素布局（行优先排列，0 基索引；换算成 1 基索引即：
    第 1~3 像素为红绿蓝标记，第 4 像素为头信息，
    第 5~5+b-1 为进制颜色表，第 5+b~5+b+g-1 为文件名，之后为正文）:

    索引 0        : (255,  0,  0)  红色标记
    索引 1        : (  0,255,  0)  绿色标记
    索引 2        : (  0,  0,255)  蓝色标记
    索引 3        : (  r,  g,  b)  头信息
                      r = 正文校验和 (sum(body) % 256)   ← 自定义用途：数据完整性校验
                      g = 文件名长度（UTF-8 字节数）
                      b = 进制（2..255）
    4 .. 4+b-1    : 进制颜色表，数字 d 的颜色 = 彩虹色
                    （hue = d/b*360°，全饱和全亮，共 b 个像素）
    4+b .. 4+b+g-1: 文件名，UTF-8 每个字节一个像素，按 256 级彩虹上色
    4+b+g .. 结束 : 正文，每个字节转成 b 进制、高位补零到固定位数，
                    每个"位数"一个像素，颜色取自颜色表（彩虹色）
    末尾不够一整行的像素用 (255,0,255) 品红填充；
    解码时正文区域遇到第一个"不在颜色表里"的像素即视为正文结束。

    彩虹色映射保证互不重复（相邻色相步进 ≥ 约 1.4°，RGB 步进 ≥ 6），
    因此任意进制 2..255 都能无损反查；白色 (255,255,255) 不在彩虹色域内
    （全饱和全亮的彩虹色必含一个 0 通道），正好用作填充结束标记。

进制 b 决定每个字节占几个像素：digits = ceil(log_b(256))，
例如 b=16 → 每字节 2 像素，b=8 → 3 像素，b=2 → 8 像素。

依赖：优先使用 Pillow；未安装时自动退回内置 zlib/struct 实现的极简 PNG 编解码。
"""

import argparse
import math
import os
import struct
import sys
import zlib
from pathlib import Path

MAGIC_RED = (255, 0, 0)
MAGIC_GREEN = (0, 255, 0)
MAGIC_BLUE = (0, 0, 255)
PAD_COLOR = (255, 255, 255)  # 白色：不在彩虹色域内（彩虹色必含一个 0 通道），用作正文结束标记
DEFAULT_BASE = 16

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


# ---------------------------------------------------------------- 进制工具
def digits_per_byte(base: int) -> int:
    """一个字节用 base 进制表示需要几位（高位补零）。"""
    d, p = 1, base
    while p < 256:
        p *= base
        d += 1
    return d


def to_base_digits(value: int, base: int, digits: int) -> list:
    """把 0..255 的字节转成 base 进制、长度固定为 digits 的数字列表（高位在前）。"""
    out = []
    for _ in range(digits):
        out.append(value % base)
        value //= base
    out.reverse()
    return out


def rainbow_color(value: int, n: int) -> tuple:
    """把 value (0..n-1) 映射为彩虹色：hue = value/n*360°，全饱和全亮。

    相邻 value 的色相步进 = 360/n ≥ 1.4°，每个 60° 扇区内 RGB 线性步进
    ≥ 约 6 个色阶，因此 n ≤ 256 时任意两个 value 的颜色都互不重复，
    可用于无损反查。
    """
    h = value * 360.0 / n
    if h < 60:
        return (255, int(round(255 * h / 60.0)), 0)
    if h < 120:
        return (int(round(255 * (120 - h) / 60.0)), 255, 0)
    if h < 180:
        return (0, 255, int(round(255 * (h - 120) / 60.0)))
    if h < 240:
        return (0, int(round(255 * (240 - h) / 60.0)), 255)
    if h < 300:
        return (int(round(255 * (h - 240) / 60.0)), 0, 255)
    return (255, 0, int(round(255 * (360 - h) / 60.0)))


# ---------------------------------------------------------------- PNG 图像层
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png_std(width: int, height: int, pixels: list) -> bytes:
    """极简 PNG 编码：8bit RGB、无隔行、每行 filter 0。pixels 长度必须 == width*height。"""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n"
    out += _png_chunk(b"IHDR", ihdr)
    out += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += _png_chunk(b"IEND", b"")
    return out


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> bytes:
    """还原 PNG 的 5 种 filter（0-4），返回逐行 RGB 字节流。"""
    stride = width * channels
    out = bytearray()
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        f = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if f == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:  # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif f != 0:
            raise ValueError(f"不支持的 PNG filter 类型: {f}")
        out.extend(line)
        prev = line
    return bytes(out)


def read_png_std(data: bytes) -> tuple:
    """极简 PNG 解码：仅支持 8bit、无隔行、color type 2(RGB)/6(RGBA)。返回 (w, h, [(r,g,b),...])。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG 文件")
    pos = 8
    width = height = None
    color_type = None
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            w, h, bitdepth, ct, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            if bitdepth != 8 or comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("仅支持 8bit、无隔行的 PNG")
            if ct not in (2, 6):
                raise ValueError("仅支持 RGB / RGBA 的 PNG")
            width, height, color_type = w, h, ct
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
        pos += 12 + length
    if width is None or height is None or not idat:
        raise ValueError("PNG 缺少 IHDR/IDAT")

    channels = 3 if color_type == 2 else 4
    raw = _unfilter(zlib.decompress(idat), width, height, channels)
    pixels = []
    for i in range(0, len(raw), channels):
        pixels.append((raw[i], raw[i + 1], raw[i + 2]))  # RGBA 时丢弃 alpha
    return width, height, pixels


def save_image(path: Path, width: int, height: int, pixels: list) -> None:
    if HAVE_PIL:
        img = Image.new("RGB", (width, height))
        img.putdata(pixels)
        img.save(path, format="PNG")
    else:
        path.write_bytes(write_png_std(width, height, pixels))


def load_pixels(path: Path) -> tuple:
    """返回 (width, height, [(r,g,b),...])。"""
    if HAVE_PIL:
        img = Image.open(path)
        img = img.convert("RGB")
        return img.width, img.height, list(img.getdata())
    w, h, pixels = read_png_std(path.read_bytes())
    return w, h, pixels


# ---------------------------------------------------------------- 编码 / 解码
def encode_file(src: Path, dst: Path, base: int, width: int) -> None:
    data = src.read_bytes()
    name = src.name.encode("utf-8")
    if not name:
        raise ValueError("文件名不能为空")
    if len(name) > 255:
        raise ValueError("文件名过长：UTF-8 字节数不能超过 255")
    if not (2 <= base <= 255):
        raise ValueError("进制 b 必须在 2..255 之间")
    dpb = digits_per_byte(base)

    # 头：红绿蓝标记 + (校验和, 文件名长度, 进制)
    pixels = [
        MAGIC_RED,
        MAGIC_GREEN,
        MAGIC_BLUE,
        (sum(data) % 256, len(name), base),
    ]
    # 进制颜色表（彩虹色）
    pixels += [rainbow_color(d, base) for d in range(base)]
    # 文件名（UTF-8 每字节一个像素，256 级彩虹）
    pixels += [rainbow_color(v, 256) for v in name]
    # 正文（b 进制，每字节固定位数，颜色取自颜色表）
    body_digits = []
    for byte in data:
        body_digits.extend(to_base_digits(byte, base, dpb))
    pixels += [rainbow_color(d, base) for d in body_digits]

    if width is None:
        width = max(1, math.ceil(math.sqrt(len(pixels))))
    height = math.ceil(len(pixels) / width)
    # 末尾不足一整行 → 品红填充（解码时第一个"非颜色表"像素即正文结束）
    pixels += [PAD_COLOR] * (width * height - len(pixels))

    save_image(dst, width, height, pixels)
    print(
        f"编码完成: {dst}  ({width}x{height}, 进制={base}, "
        f"文件名={src.name!r}({len(name)}字节), 正文={len(data)}字节 → {len(body_digits)}像素)"
    )


def decode_file(img: Path, out_dir: Path) -> None:
    width, height, pixels = load_pixels(img)
    if len(pixels) < 4:
        raise ValueError("图片太小，不是有效的 fpic 图片")
    if (pixels[0], pixels[1], pixels[2]) != (MAGIC_RED, MAGIC_GREEN, MAGIC_BLUE):
        raise ValueError("缺少红绿蓝标记，不是 fpic 图片")

    checksum, name_len, base = pixels[3]
    if not (2 <= base <= 255):
        raise ValueError(f"非法进制: {base}")
    if name_len > len(pixels) - 4 - base:
        raise ValueError("头信息中的文件名长度超出图片范围")

    # 颜色表 → 数字映射
    table = pixels[4:4 + base]
    digit_map = {}
    for d, c in enumerate(table):
        if c in digit_map:
            raise ValueError(f"进制颜色表存在重复颜色: {c}")
        digit_map[c] = d

    # 文件名（256 级彩虹反查表）
    start = 4 + base
    name_map = {rainbow_color(v, 256): v for v in range(256)}
    name_px = pixels[start:start + name_len]
    try:
        name_bytes = bytes(name_map[px] for px in name_px)
    except KeyError:
        raise ValueError("文件名区域包含无法识别的颜色，图片可能已损坏")

    # 正文：扫描到第一个不在颜色表里的像素为止
    body_digits = []
    idx = start + name_len
    while idx < len(pixels):
        c = pixels[idx]
        if c not in digit_map:
            break
        body_digits.append(digit_map[c])
        idx += 1

    dpb = digits_per_byte(base)
    if len(body_digits) % dpb != 0:
        raise ValueError(
            f"正文数字数 {len(body_digits)} 不是每字节位数 {dpb} 的倍数，图片可能被截断或损坏"
        )

    out = bytearray()
    for i in range(0, len(body_digits), dpb):
        v = 0
        for dd in body_digits[i:i + dpb]:
            v = v * base + dd
        out.append(v & 0xFF)

    actual = sum(out) % 256
    if actual != checksum:
        print(
            f"警告: 校验和不匹配！期望 {checksum}，实际 {actual}，数据可能已损坏",
            file=sys.stderr,
        )

    try:
        fname = name_bytes.decode("utf-8")
    except UnicodeDecodeError:
        fname = name_bytes.decode("latin-1")
    fname = os.path.basename(fname) or "decoded.bin"  # 防路径穿越
    dst = out_dir / fname
    dst.write_bytes(bytes(out))
    print(f"解码完成: {dst}  ({len(out)} 字节，文件名来自图片头)")


# ---------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="fpic —— 文件 ⇄ 图片互转（PNG 无损）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="文件 → 图片")
    pe.add_argument("input", type=Path, help="输入文件")
    pe.add_argument("output", type=Path, help="输出 PNG 路径")
    pe.add_argument("--width", type=int, default=None,
                    help="图片宽度（默认自动取近似正方形）")
    pe.add_argument("--base", type=int, default=DEFAULT_BASE,
                    help=f"进制 b，2..255（默认 {DEFAULT_BASE}）")

    pd = sub.add_parser("decode", help="图片 → 文件")
    pd.add_argument("input", type=Path, help="输入 PNG 路径")
    pd.add_argument("--out-dir", type=Path, default=Path("."),
                    help="输出目录（默认当前目录，文件名取自图片头）")

    args = parser.parse_args()

    if args.cmd == "encode":
        if not args.input.is_file():
            sys.exit(f"输入文件不存在: {args.input}")
        if args.width is not None and args.width < 1:
            sys.exit("宽度必须 ≥ 1")
        try:
            encode_file(args.input, args.output, args.base, args.width)
        except ValueError as e:
            sys.exit(f"编码失败: {e}")
    else:
        if not args.input.is_file():
            sys.exit(f"输入图片不存在: {args.input}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            decode_file(args.input, args.out_dir)
        except ValueError as e:
            sys.exit(f"解码失败: {e}")


if __name__ == "__main__":
    main()
