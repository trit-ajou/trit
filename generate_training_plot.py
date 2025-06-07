#!/usr/bin/env python3
"""
훈련 완료 후 JSON 데이터로부터 시각화 그래프를 생성하는 스크립트
"""

import json
import matplotlib.pyplot as plt
import os

def generate_plot_from_json(json_path):
    """JSON 파일로부터 훈련 시각화 그래프 생성"""
    
    # JSON 데이터 로드
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    
    # 그래프 생성
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 1: 전체 손실 비교
    plt.subplot(2, 2, 1)
    if train_losses:
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 로그 스케일
    plt.subplot(2, 2, 2)
    if train_losses:
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 손실 차이
    plt.subplot(2, 2, 3)
    if train_losses and val_losses and len(train_losses) == len(val_losses):
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_diff, 'g-', label='Val - Train Loss', linewidth=2, marker='^', markersize=3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.title('Validation - Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 서브플롯 4: 통계 정보
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 통계 값들을 미리 계산
    final_train = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
    final_val = f"{val_losses[-1]:.4f}" if val_losses else "N/A"
    best_val = f"{min(val_losses):.4f}" if val_losses else "N/A"
    best_epoch = epochs[val_losses.index(min(val_losses))] if val_losses else "N/A"
    
    if train_losses and len(train_losses) > 1:
        loss_reduction = f"{((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%"
    else:
        loss_reduction = "N/A"
    
    stats_text = f"""
Training Summary:
• Total Epochs: {len(epochs)}
• Final Train Loss: {final_train}
• Final Val Loss: {final_val}
• Best Val Loss: {best_val}
• Best Val Epoch: {best_epoch}
• Loss Reduction: {loss_reduction}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 그래프 저장
    save_dir = os.path.dirname(json_path)
    plot_path = os.path.join(save_dir, "final_training_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training analysis plot saved to: {plot_path}")
    
    # 훈련 결과 요약 출력
    print("\n🎉 Training Results Summary:")
    print(f"   📊 Total Epochs: {len(epochs)}")
    print(f"   📉 Final Train Loss: {final_train}")
    print(f"   📉 Final Val Loss: {final_val}")
    print(f"   🏆 Best Val Loss: {best_val} (Epoch {best_epoch})")
    print(f"   📈 Loss Reduction: {loss_reduction}")

if __name__ == "__main__":
    # JSON 파일 경로
    json_path = "trit/models/lora/training_losses.json"
    
    if os.path.exists(json_path):
        generate_plot_from_json(json_path)
    else:
        print(f"❌ JSON file not found: {json_path}")
        print("Please check the path or run training first.")
