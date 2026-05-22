from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from trusted_cl_nids.data import load_dataset
from trusted_cl_nids.trainer import TrainerConfig, TrustedPseudoLabelCLTrainer
from trusted_cl_nids.utils import resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trusted pseudo-label continual learning prototype for NIDS."
    )
    project_root = Path(__file__).resolve().parent
    local_data_root = project_root
    fallback_data_root = project_root.parent / "code" / "SSF"
    default_data_root = (
        local_data_root
        if (local_data_root / "NSL_pre_data" / "PKDDTrain+.csv").exists()
        else fallback_data_root
    )
    parser.add_argument("--dataset", default="nsl", choices=["nsl"])
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=5011)

    parser.add_argument("--initial-ratio", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--initial-epochs", type=int, default=5)
    parser.add_argument("--online-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--consistency-tau", type=float, default=0.90)
    parser.add_argument("--noise-std", type=float, default=0.01)

    parser.add_argument("--memory-size", type=int, default=4096)
    parser.add_argument("--replay-batch-size", type=int, default=256)
    parser.add_argument("--drift-alpha", type=float, default=0.05)
    parser.add_argument("--reference-size", type=int, default=5000)
    parser.add_argument("--always-update", action="store_true")
    parser.add_argument("--oracle-budget", type=int, default=0)

    parser.add_argument("--lambda-recon", type=float, default=0.2)
    parser.add_argument("--lambda-contrastive", type=float, default=0.05)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--lambda-distill", type=float, default=0.2)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-weight", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "outputs" / f"{args.dataset}_{run_name}")

    bundle = load_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        initial_ratio=args.initial_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples or None,
        max_test_samples=args.max_test_samples or None,
    )
    config = TrainerConfig(
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        online_epochs=args.online_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        max_windows=args.max_windows,
        tau=args.tau,
        consistency_tau=args.consistency_tau,
        noise_std=args.noise_std,
        memory_size=args.memory_size,
        replay_batch_size=args.replay_batch_size,
        drift_alpha=args.drift_alpha,
        reference_size=args.reference_size,
        lambda_recon=args.lambda_recon,
        lambda_contrastive=args.lambda_contrastive,
        lambda_consistency=args.lambda_consistency,
        lambda_distill=args.lambda_distill,
        replay_weight=args.replay_weight,
        pseudo_weight=args.pseudo_weight,
        always_update=args.always_update,
        oracle_budget=args.oracle_budget,
        output_dir=output_dir,
    )
    print(f"dataset={bundle.dataset_name} input_dim={bundle.input_dim}")
    print(f"initial={len(bundle.y_initial)} stream={len(bundle.y_stream)} test={len(bundle.y_test)}")
    print(f"device={device} output_dir={output_dir}")

    trainer = TrustedPseudoLabelCLTrainer(bundle, config, device)
    final_metrics = trainer.run()
    print("final_metrics")
    for key in sorted(final_metrics):
        print(f"{key}: {final_metrics[key]}")


if __name__ == "__main__":
    main()
