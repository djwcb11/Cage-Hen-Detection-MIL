> > 📢 **Official Implementation Notice**  
> > This repository contains the official implementation of the paper:  
> > **"Low-Yield Hen Identification in Stacked Cages via Head Detection and Prototype-Based Multiple Instance Learning"**  
> > *Submitted to **The Visual Computer** (Springer), 2026.*  
> >
> > 🔖 If you find this work useful in your research, **please consider citing our paper**  🙏
> >
> > ---
>
> ## 📖 Abstract
>
> Precision poultry farming relies on efficient automated screening of low-yield laying hens, especially in multi-tier stacked cage systems. Manual inspection is inefficient, and existing deep learning methods depend on costly instance-level annotations. This work proposes a two-stage visual computing framework combining target detection and weakly supervised multiple instance learning for cage-level low-yield hen identification. An improved lightweight YOLOv8n with SEAM attention, SCDown downsampling, and RepHead is designed to locate hen heads accurately at 89.66\% mAP and 418.8 FPS. A prototype-guided multiple instance learning model ProtoMIL then achieves cage-level classification with 81.43\% AUC and 73.18\% accuracy under weak supervision. This framework reduces annotation costs and supports real-time edge deployment, providing a practical solution for intelligent poultry farming with broad applicability for visual computing in agricultural scenarios.
>
> ---
>
> ## ✨ Highlights
>
> Two-stage framework identifies low-yield hens at the cage level.
> Improved YOLOv8n extracts hen heads accurately in dense cages.
> ProtoMIL classifies low-yield cages using weak bag-level labels.
> Global prototype strategy overcomes few-shot challenges in MIL.
>
> ### Requirements
>
> - Python ≥ 3.10
> - PyTorch ≥ 1.13
> - CUDA ≥ 11.7
