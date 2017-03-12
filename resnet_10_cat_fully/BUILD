package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//resnet/...",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)

py_library(
    name = "resnet_model",
    srcs = ["resnet_model.py"],
    srcs_version = PY3,
)

py_binary(
    name = "resnet_main",
    srcs = [
        "resnet_main.py",
    ],
    deps = [
        ":cifar_input",
        ":resnet_model",
    ],
    default_python_version = PY3,
    srcs_version = PY3,
)

py_library(
    name = "cifar_input",
    srcs = ["cifar_input.py"],
    srcs_version = PY3,
)
