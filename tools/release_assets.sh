#!/bin/bash
# Build the release archive on a Mac: CLI, Metal library,
# licence files, and a sha256 manifest, under dist/. Run at the tagged commit:
#   bash tools/release_assets.sh v0.1.0
# Refuses a dirty tree so the archive matches the tag.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag, e.g. v0.1.0}
[ "$(uname -s)" = Darwin ] || { echo "run on macOS (Metal)"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "tree is dirty; commit or stash first"; exit 1; }
TAG_COMMIT=$(git rev-parse --verify "refs/tags/$TAG^{commit}")
[ "$TAG_COMMIT" = "$(git rev-parse HEAD)" ] || { echo "HEAD does not match tag $TAG"; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
NAME="zdraw-${TAG}-macos-arm64"
source tools/release_flags.sh
zig build "${ZIG_RELEASE_FLAGS[@]}" -Doptimize=ReleaseFast -Dversion="${TAG#v}" -Dcommit="$COMMIT"
rm -rf "dist/$NAME" && mkdir -p "dist/$NAME/lib"
cp zig-out/bin/zdraw "dist/$NAME/"
# The steel metallib is the default attention/GEMM route: without it a user
# without Xcode silently ran MFA. It ships beside the binary (lib/, where the
# loader looks first after ZDRAW_STEEL_LIB).
[ -f zig-out/lib/steel.metallib ] || { echo "zig-out/lib/steel.metallib missing (Xcode metal toolchain?)"; exit 1; }
cp zig-out/lib/steel.metallib "dist/$NAME/lib/"
cp LICENSE NOTICE CHANGELOG.md COMMERCIAL.md SUPPORT.md "dist/$NAME/"
cp docs/release-readme.md "dist/$NAME/README.md"
mkdir -p "dist/$NAME/vendor/mfa" "dist/$NAME/vendor/steel" "dist/$NAME/licenses/stanza"
cp vendor/mfa/LICENSE* "dist/$NAME/vendor/mfa/"
cp vendor/steel/LICENSE "dist/$NAME/vendor/steel/"
cp zig-out/share/licenses/stanza/LICENSE "dist/$NAME/licenses/stanza/"
# Developer ID signing when the identity is given (hardened runtime, timestamped).
if [ -n "${SIGN_IDENTITY:-}" ]; then
  /usr/bin/codesign --sign "$SIGN_IDENTITY" --options runtime --timestamp --force "dist/$NAME/zdraw"
fi
{
  echo "zdraw $TAG"; echo "commit $COMMIT"; echo "zig $(zig version)"
  echo "built $(date -u +%Y-%m-%dT%H:%MZ) on $(sysctl -n machdep.cpu.brand_string) macOS $(sw_vers -productVersion)"
} > "dist/$NAME/BUILD.txt"
# COPYFILE_DISABLE keeps macOS tar from adding AppleDouble ._ sidecars for extended attributes.
( cd dist && COPYFILE_DISABLE=1 tar -czf "$NAME.tar.gz" "$NAME" && shasum -a 256 "$NAME.tar.gz" > "$NAME.sha256" )
( cd "dist/$NAME" && shasum -a 256 zdraw lib/steel.metallib ) > "dist/$NAME.manifest.sha256"
python3 tools/release_smoke.py "dist/$NAME.tar.gz"
echo "RELEASE-ASSETS dist/$NAME.tar.gz"; cat "dist/$NAME.sha256"
