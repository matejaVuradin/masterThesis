import torch
from tqdm import tqdm
import numpy as np

from .config import CycleGANConfig
from .metrics import calculate_metrics, calculate_metrics_fixed_crop

def test_model(G_AB, G_BA, test_dataloader, config: CycleGANConfig):
    G_AB.eval()
    G_BA.eval()
    
    # Metrike
    metrics_t1_to_t2 = {"ssim": [], "mse": [], "psnr": [], "ssim_crop": [], "mse_crop": [], "psnr_crop": []}
    metrics_t2_to_t1 = {"ssim": [], "mse": [], "psnr": [], "ssim_crop": [], "mse_crop": [], "psnr_crop": []}
    metrics_cycle_t1 = {"ssim": [], "mse": [], "psnr": [], "ssim_crop": [], "mse_crop": [], "psnr_crop": []}
    metrics_cycle_t2 = {"ssim": [], "mse": [], "psnr": [], "ssim_crop": [], "mse_crop": [], "psnr_crop": []}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluacija modela"):
            real_A = batch["A"].to(config.device)  # T1
            real_B = batch["B"].to(config.device)  # T2
            
            # translacije
            fake_B = G_AB(real_A)  # T1 -> T2
            fake_A = G_BA(real_B)  # T2 -> T1
            
            # rekonstrukcije
            rec_A = G_BA(fake_B)  # T1 -> T2 -> T1
            rec_B = G_AB(fake_A)  # T2 -> T1 -> T2
            
            # normaliziraj
            real_A_norm = (real_A * 0.5 + 0.5)
            real_B_norm = (real_B * 0.5 + 0.5)
            fake_A_norm = (fake_A * 0.5 + 0.5)
            fake_B_norm = (fake_B * 0.5 + 0.5)
            rec_A_norm = (rec_A * 0.5 + 0.5)
            rec_B_norm = (rec_B * 0.5 + 0.5)
            
            # Izračunaj metrike za svaku sliku u batchu
            for i in range(real_A.size(0)):
                # T1 -> T2 translacija
                metrics = calculate_metrics(real_B_norm[i].squeeze(), fake_B_norm[i].squeeze())
                metrics_t1_to_t2["ssim"].append(metrics["ssim"])
                metrics_t1_to_t2["mse"].append(metrics["mse"])
                metrics_t1_to_t2["psnr"].append(metrics["psnr"])
                
                # T2 -> T1 translacija
                metrics = calculate_metrics(real_A_norm[i].squeeze(), fake_A_norm[i].squeeze())
                metrics_t2_to_t1["ssim"].append(metrics["ssim"])
                metrics_t2_to_t1["mse"].append(metrics["mse"])
                metrics_t2_to_t1["psnr"].append(metrics["psnr"])
                
                # T1 -> T2 -> T1 ciklus
                metrics = calculate_metrics(real_A_norm[i].squeeze(), rec_A_norm[i].squeeze())
                metrics_cycle_t1["ssim"].append(metrics["ssim"])
                metrics_cycle_t1["mse"].append(metrics["mse"])
                metrics_cycle_t1["psnr"].append(metrics["psnr"])
                
                # T2 -> T1 -> T2 ciklus
                metrics = calculate_metrics(real_B_norm[i].squeeze(), rec_B_norm[i].squeeze())
                metrics_cycle_t2["ssim"].append(metrics["ssim"])
                metrics_cycle_t2["mse"].append(metrics["mse"])
                metrics_cycle_t2["psnr"].append(metrics["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_B_norm[i].squeeze(), fake_B_norm[i].squeeze())
                metrics_t1_to_t2["ssim_crop"].append(metrics_crop["ssim"])
                metrics_t1_to_t2["mse_crop"].append(metrics_crop["mse"])
                metrics_t1_to_t2["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_A_norm[i].squeeze(), fake_A_norm[i].squeeze())
                metrics_t2_to_t1["ssim_crop"].append(metrics_crop["ssim"])
                metrics_t2_to_t1["mse_crop"].append(metrics_crop["mse"])
                metrics_t2_to_t1["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_A_norm[i].squeeze(), rec_A_norm[i].squeeze())
                metrics_cycle_t1["ssim_crop"].append(metrics_crop["ssim"])
                metrics_cycle_t1["mse_crop"].append(metrics_crop["mse"])
                metrics_cycle_t1["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_B_norm[i].squeeze(), rec_B_norm[i].squeeze())
                metrics_cycle_t2["ssim_crop"].append(metrics_crop["ssim"])
                metrics_cycle_t2["mse_crop"].append(metrics_crop["mse"])
                metrics_cycle_t2["psnr_crop"].append(metrics_crop["psnr"])

    
    # prosječne metrike
    avg_metrics = {
        "T1->T2": {
            "ssim": np.mean(metrics_t1_to_t2["ssim"]),
            "mse": np.mean(metrics_t1_to_t2["mse"]),
            "psnr": np.mean(metrics_t1_to_t2["psnr"])
        },
        "T2->T1": {
            "ssim": np.mean(metrics_t2_to_t1["ssim"]),
            "mse": np.mean(metrics_t2_to_t1["mse"]),
            "psnr": np.mean(metrics_t2_to_t1["psnr"])
        },
        "T1->T2->T1": {
            "ssim": np.mean(metrics_cycle_t1["ssim"]),
            "mse": np.mean(metrics_cycle_t1["mse"]),
            "psnr": np.mean(metrics_cycle_t1["psnr"])
        },
        "T2->T1->T2": {
            "ssim": np.mean(metrics_cycle_t2["ssim"]),
            "mse": np.mean(metrics_cycle_t2["mse"]),
            "psnr": np.mean(metrics_cycle_t2["psnr"])
        }
    }

    agv_metrics_crop = {
        "T1->T2": {
            "ssim": np.mean(metrics_t1_to_t2["ssim_crop"]),
            "mse": np.mean(metrics_t1_to_t2["mse_crop"]),
            "psnr": np.mean(metrics_t1_to_t2["psnr_crop"])
        },
        "T2->T1": {
            "ssim": np.mean(metrics_t2_to_t1["ssim_crop"]),
            "mse": np.mean(metrics_t2_to_t1["mse_crop"]),
            "psnr": np.mean(metrics_t2_to_t1["psnr_crop"])
        },
        "T1->T2->T1": {
            "ssim": np.mean(metrics_cycle_t1["ssim_crop"]),
            "mse": np.mean(metrics_cycle_t1["mse_crop"]),
            "psnr": np.mean(metrics_cycle_t1["psnr_crop"])
        },
        "T2->T1->T2": {
            "ssim": np.mean(metrics_cycle_t2["ssim_crop"]),
            "mse": np.mean(metrics_cycle_t2["mse_crop"]),
            "psnr": np.mean(metrics_cycle_t2["psnr_crop"])
        }
    }

    print("\n=== Rezultati evaluacije ===")
    print("T1 -> T2 translacija:")
    print(f"  SSIM: {avg_metrics['T1->T2']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T1->T2']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T1->T2']['psnr']:.2f} dB")
    
    print("\nT2 -> T1 translacija:")
    print(f"  SSIM: {avg_metrics['T2->T1']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T2->T1']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T2->T1']['psnr']:.2f} dB")
    
    print("\nT1 -> T2 -> T1 rekonstrukcija:")
    print(f"  SSIM: {avg_metrics['T1->T2->T1']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T1->T2->T1']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T1->T2->T1']['psnr']:.2f} dB")
    
    print("\nT2 -> T1 -> T2 rekonstrukcija:")
    print(f"  SSIM: {avg_metrics['T2->T1->T2']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T2->T1->T2']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T2->T1->T2']['psnr']:.2f} dB")

    print("\n=== Rezultati evaluacije (fiksni crop) ===")
    print("T1 -> T2 translacija:")
    print(f"  SSIM: {agv_metrics_crop['T1->T2']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T1->T2']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T1->T2']['psnr']:.2f} dB")
    print("\nT2 -> T1 translacija:")
    print(f"  SSIM: {agv_metrics_crop['T2->T1']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T2->T1']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T2->T1']['psnr']:.2f} dB")
    print("\nT1 -> T2 -> T1 rekonstrukcija:")
    print(f"  SSIM: {agv_metrics_crop['T1->T2->T1']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T1->T2->T1']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T1->T2->T1']['psnr']:.2f} dB")
    print("\nT2 -> T1 -> T2 rekonstrukcija:")
    print(f"  SSIM: {agv_metrics_crop['T2->T1->T2']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T2->T1->T2']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T2->T1->T2']['psnr']:.2f} dB")