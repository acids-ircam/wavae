from os import system, makedirs

VERSION = input("Version: ")
PACKAGE = f"wavae_{VERSION}"

makedirs(f"{PACKAGE}/usr/lib/")
makedirs(f"{PACKAGE}/usr/local/lib/pd-externals/wavae/")
makedirs(f"{PACKAGE}/DEBIAN/")

system(f"cp build/*.pd_linux {PACKAGE}/usr/local/lib/pd-externals/wavae/")
system(f"cp helppatch.pd {PACKAGE}/usr/local/lib/pd-externals/wavae/help-encoder~.pd")
system(f"cp helppatch.pd {PACKAGE}/usr/local/lib/pd-externals/wavae/help-decoder~.pd")

system(f"cp build/libwavae/libwavae.so {PACKAGE}/usr/lib/")

with open(f"{PACKAGE}/DEBIAN/control", "w") as control:
    control.write("Package: wavae\n")
    control.write(f"Version: {VERSION}\n")
    control.write("Maintainer: Antoine CAILLON <caillon@ircam.fr>\n")
    control.write("Depends: nvidia-cuda-toolkit\n")
    control.write("Architecture: all\n")
    control.write(
        "Description: WaVAE puredata external. Needs libtorch in /usr/lib\n")

system(f"dpkg-deb --build {PACKAGE}")
if not input("Enter any key to prevent temporary folder destruction: "):
    system(f"rm -fr {PACKAGE}/")
