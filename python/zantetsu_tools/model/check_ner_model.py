"""Inspect the NER model weights and label mapping."""

from __future__ import annotations

from ..common.safetensors_tools import inspect_ner_model


def main() -> None:
    report = inspect_ner_model()
    print(f"classifier.weight shape: {tuple(report['classifier_weight_shape'])}")
    print(f"classifier.bias shape: {tuple(report['classifier_bias_shape'])}")
    print(f"Number of labels in model: {report['num_labels']}")
    print()

    print("config.json id2label (ordered by index):")
    for key, value in sorted(report["id2label"].items(), key=lambda item: int(item[0])):
        print(f"  {key}: {value}")
    print()

    print("Expected training tag order:")
    for index, tag in enumerate(report["train_tags"]):
        print(f"  {index}: {tag}")
    print()

    print("Cross-checking config.json vs training tags:")
    if not report["mismatches"]:
        print("  No mismatches found")
    else:
        for mismatch in report["mismatches"]:
            print(
                "  Index {index}: expected='{expected}' vs actual='{actual}'".format(
                    **mismatch
                )
            )

    print()
    print("All safetensor keys:")
    for key in report["keys"]:
        print(f"  {key}")


if __name__ == "__main__":
    main()
