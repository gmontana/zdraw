#include <CoreGraphics/CoreGraphics.h>
#include <Foundation/Foundation.h>
#include <ImageIO/ImageIO.h>
#include <stdint.h>
#include <stdlib.h>

int zdraw_image_decode_rgba8(
    const char *path,
    uint8_t **out_pixels,
    uint32_t *out_width,
    uint32_t *out_height
) {
    if (path == NULL || out_pixels == NULL || out_width == NULL || out_height == NULL) {
        return -1;
    }

    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        if (ns_path == nil) return -2;

        NSURL *url = [NSURL fileURLWithPath:ns_path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (source == NULL) return -3;

        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (image == NULL) return -4;

        const size_t width = CGImageGetWidth(image);
        const size_t height = CGImageGetHeight(image);
        if (width == 0 || height == 0 || width > UINT32_MAX || height > UINT32_MAX) {
            CGImageRelease(image);
            return -5;
        }
        if (width > SIZE_MAX / 4 || height > SIZE_MAX / (width * 4)) {
            CGImageRelease(image);
            return -6;
        }

        const size_t row_bytes = width * 4;
        uint8_t *pixels = (uint8_t *)calloc(height, row_bytes);
        if (pixels == NULL) {
            CGImageRelease(image);
            return -7;
        }

        CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
        if (color_space == NULL) {
            free(pixels);
            CGImageRelease(image);
            return -8;
        }

        CGContextRef ctx = CGBitmapContextCreate(
            pixels,
            width,
            height,
            8,
            row_bytes,
            color_space,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );
        CGColorSpaceRelease(color_space);
        if (ctx == NULL) {
            free(pixels);
            CGImageRelease(image);
            return -9;
        }

        CGContextClearRect(ctx, CGRectMake(0, 0, (CGFloat)width, (CGFloat)height));
        CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)width, (CGFloat)height), image);
        CGContextRelease(ctx);
        CGImageRelease(image);

        *out_pixels = pixels;
        *out_width = (uint32_t)width;
        *out_height = (uint32_t)height;
        return 0;
    }
}

void zdraw_image_free(void *ptr) {
    free(ptr);
}
