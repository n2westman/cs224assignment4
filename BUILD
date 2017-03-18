py_library(
    name = "model",
    srcs = [
        "contrib_ops.py",
        "data_utils.py",
        "evaluate.py",
        "qa_data.py",
        "qa_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/py/tqdm",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/python",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":model",
        "//learning/brain/public:tensorflow_gpu_deps",
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)
