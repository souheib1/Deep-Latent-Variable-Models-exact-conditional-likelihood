# Deep Latent Variable Models: Exact Likelihood

Missing data imputation using the exact conditional likelihood of DLVM

### Some Results : 

We're comparing the sampling methods to fill in missing parts of images by looking at the original images alongside versions where some fraction of pixels have been randomly changed.
1. For MNIST dataset:

![comparison_40_random](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/19affe32-e33e-4696-af51-ae723a585e0a)
![comparison_50_half_top](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/7401728f-be44-4ec3-83ff-3f5bccfd2a2a)

2. For OMNIGLOT dataset:
   
![comparison_50_random](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/39952121-1a21-4dc5-8605-ac28b342be64)

Here we show some results about the imputation performance for different percentages of missing data. 


![F1evolution-Top](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/d8c4b090-3ede-48d4-905e-a827e0cad99f)
Finally, we challenge the model by different initializations of the missing parts and we explore the the imputation performance. 

![Noise_comparaison](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/4efce873-5bca-4071-adc3-45ca86a27074)
![Initialization](https://github.com/souheib1/Deep-Latent-Variable-Models-exact-conditional-likelihood/assets/73786465/9ff0cbf6-2dc4-493c-bd65-21eefe3570ac)


## References
```
@misc{mattei2018leveraging,
      title={Leveraging the Exact Likelihood of Deep Latent Variable Models}, 
      author={Pierre-Alexandre Mattei and Jes Frellsen},
      year={2018},
      eprint={1802.04826},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}

@misc{kingma2022autoencoding,
      title={Auto-Encoding Variational Bayes}, 
      author={Diederik P Kingma and Max Welling},
      year={2022},
      eprint={1312.6114},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
