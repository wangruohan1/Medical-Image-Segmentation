# Medical Iamge Segmentation、Uncertainty Estimation  
该站点整理了“医学图像分割、不确定性相关”方向的论文、代码、博客等学习资源

# 目录
- [医学图像分割](#1医学图像分割)  
   - [基础backbone](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/backbone)   
   - [不同医学场景下的图像分割](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Medical_Image_Segmentation)
- [不确定性估计](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Uncertainty_Estimation)
   - [贝叶斯方法](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Uncertainty_Estimation/Bayesian)
   - [集成方法](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Uncertainty_Estimation/Ensemble)
   - [证据深度学习方法](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Uncertainty_Estimation/Evidential%20Deep%20learning)
- [证据深度学习在医学图像分割中的应用](https://github.com/wangruohan1/Medical-Image-Segmentation/tree/master/Uncertainty_Estimation/EDL%20%E7%94%A8%E4%BA%8E%E5%8C%BB%E5%AD%A6%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2)
- [代码和数据集](#二代码和数据集)
- [博客](#三博客)
   - [医学图像分割](#1医学图像分割)
   - [证据深度学习](#2证据深度学习)

# 一、论文
### 1.医学图像分割
#### （1）基础backbone  
- Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic
segmentation (2014), arXiv:1411.4038 [cs.CV]（FCN）
-  O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. MICCAI. Cham,
Switzerland: Springer, 2015, pp. 234–241.（UNet）
- Zhou Z, Rahman Siddiquee M M, Tajbakhsh N, et al. Unet++: A nested u-net architecture for medical image segmentation[C]//Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4. Springer International Publishing, 2018: 3-11.(UNet++)
- Qin X, Zhang Z, Huang C, et al. U2-Net: Going deeper with nested U-structure for salient object detection[J]. Pattern recognition, 2020, 106: 107404.(U2-Net)
- Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.(Deeplab_v3)
####  (2)不同医学场景下的图像分割
- Fan D P, Zhou T, Ji G P, et al. Inf-net: Automatic covid-19 lung infection segmentation from ct images[J]. IEEE Transactions on Medical Imaging, 2020, 39(8): 2626-2637.(COVID19感染区域分割)
- Fan D P, Ji G P, Zhou T, et al. Pranet: Parallel reverse attention network for polyp segmentation[C]//Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VI 23. Springer International Publishing, 2020: 263-273.（息肉分割）
### 2.不确定性估计
#### （1）综述
- Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt, M., Feng, J., ... & Zhu, X. X. (2021). A survey of uncertainty in deep neural networks. _arXiv preprint arXiv:2107.03342_.
- Abdar, M., Pourpanah, F., Hussain, S., Rezazadegan, D., Liu, L., Ghavamzadeh, M., ... & Nahavandi, S. (2021). A review of uncertainty quantification in deep learning: Techniques, applications and challenges. _Information Fusion_, _76_, 243-297.
- Uncertainty in Deep Learning（Gal博士论文）
- He, W., & Jiang, Z. (2023). A Survey on Uncertainty Quantification Methods for Deep Neural Networks: An Uncertainty Source Perspective. _arXiv preprint arXiv:2302.13425_.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. _Machine Learning_, _110_, 457-506.（数据和模型不确定性）
#### （2）贝叶斯方法：
- Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In _international conference on machine learning_ (pp. 1050-1059). PMLR.（将Dropout看做贝叶斯近似的经典论文）
- **Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. _Advances in neural information processing systems_, _30_.（不确定性估计必读论文，将不确定性分为数据不确定性以及模型不确定性，并介绍了在分类和回归中不确定性估计的建模方法）**
- Louizos, C., & Welling, M. (2017, July). Multiplicative normalizing flows for variational bayesian neural networks. In _International Conference on Machine Learning_ (pp. 2218-2227). PMLR.（变分贝叶斯神经网络，EDL论文中的对比方法）
#### （3）集成方法
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. _Advances in neural information processing systems_, _30_.（集成方法的开山之作）
#### （4）证据深度学习
- **Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. _Advances in neural information processing systems_, _31_.（证据分类）**
- Sensoy, M., Kaplan, L., Cerutti, F., & Saleki, M. (2020, April). Uncertainty-aware deep classifiers using generative models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 34, No. 04, pp. 5620-5627).（EDL作者的另一篇论文）
- Malinin, A., & Gales, M. (2018). Predictive uncertainty estimation via prior networks. _Advances in neural information processing systems_, _31_.（使用狄利克雷分布建模不确定性的另一种方法）
- Ulmer, D. (2021). A survey on evidential deep learning for single-pass uncertainty estimation. _arXiv preprint arXiv:2110.03051_.（证据不确定性综述）
### 3.证据深度学习在医学图像分割中的应用
- Zou K, Yuan X, Shen X, et al. EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap[J]. arXiv preprint arXiv:2301.00349, 2023.
- Zou K, Yuan X, Shen X, et al. TBraTS: Trusted brain tumor segmentation[C]//Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part VIII. Cham: Springer Nature Switzerland, 2022: 503-513.（可信的脑肿瘤分割）
# 二、代码和数据集
#### 1.代码
(1）[UNet]()  
(2）[FCN]()   
(3）[u2net]()   
(4) [deeplab_v3]()   
(5) [Inf-Net]()  
(6) [PraNet]()  
(7) [UNet++]()  
(8) [TBraTS-mian]()  
(9) [NMIS-main]()

#### 2.数据集
相关论文的数据集可以在其代码仓库中找到
# 三、博客
### 1.医学图像分割
- [语义分割入门1(李沐)](https://www.bilibili.com/video/BV1BK4y1M7Rd/?spm_id_from=333.999.0.0&vd_source=11905de701353d14e415365bbd180544)  
 [语义分割入门2](https://www.bilibili.com/video/BV1ev411P7dR/?spm_id_from=333.999.0.0&vd_source=11905de701353d14e415365bbd180544)  
- [FCN全卷积神经网络](https://www.bilibili.com/video/BV1af4y1L7Zu/?spm_id_from=333.999.0.0&vd_source=11905de701353d14e415365bbd180544)  
- [UNet网络结构讲解](https://www.bilibili.com/video/BV1Vq4y127fB/?spm_id_from=333.999.0.0&vd_source=11905de701353d14e415365bbd180544)  
- [DeepLabV3网络简析](https://blog.csdn.net/qq_37541097/article/details/121797301?spm=1001.2014.3001.5502)  
- [U2Net网络简介](https://blog.csdn.net/qq_37541097/article/details/126255483?spm=1001.2014.3001.5502)

### 2.证据深度学习
- [如何创造可信任的机器学习模型？先要理解不确定性 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755237&idx=3&sn=55beb3edcef0bb4ded4b56e1379efbda&chksm=871a94dbb06d1dcddc49272f77899561c0da5760f2dc6cfebd3877272a959e01c69105a8bac2#rd)