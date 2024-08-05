import gpytorch
import torch


covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(64))
# # # # # #确定RBFKernel的超参数
covar_module.base_kernel.lengthscale = 0.6 #94 93
# covar_module.base_kernel.lengthscale = 1.0 #97.0 96.6
# covar_module.base_kernel.lengthscale = 0.9 #97.4 97.1
# covar_module.base_kernel.lengthscale = 0.8 #97.3 97.4
# covar_module.base_kernel.lengthscale = 0.7 #97.2 97.0

# covar_module.base_kernel.lengthscale = 0.65 #97.2 97.1 train 99.6

# covar_module.base_kernel.lengthscale = 0.75 #97.6 97.5  train 99.6
# covar_module.base_kernel.lengthscale = 1 #97.8 97.5  train 98.7
# covar_module.base_kernel.lengthscale = 0.53 #96.1 96.0  train 97.2

# covar_module.base_kernel.lengthscale = 0.5 #97.7 97.5
# covar_module.base_kernel.lengthscale = 0.4 #94.5 94.7
# covar_module.base_kernel.lengthscale = 0.3 #95.3 94.9
# covar_module.base_kernel.lengthscale = 0.1  #95.9 95.9


def convgp(X, X2=None, global_model=None, attention_model=None):
    # Xp = get_patches(X)
    # Xp: N x num_patches x 64
    # global: N x num_patches(or 64)
    # 1.Xp+global 成为全局特征
    # 2.K(Xp)+K(global) 成为全局特征
    #
    # 局部不加全局
    #
    # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(64))
    # covar_module.base_kernel.raw_lengthscale = attention_model.gplength  # 94 93
    # print(covar_module.base_kernel.lengthscale.item())
    # print(attention_model.gplength)
    # print(attention_model.atts)
    # 确定RBFKernel的超参数
    # covar_module.base_kernel.lengthscale = 6  # 94 93
    num_patchs = X.shape[2] ** 2
    if X2 is not None:
        # Xp = X.reshape(-1, 640, 5, 5)  # 200 64 25
        # Xp2 = X2.reshape(-1, 640, 5, 5)  # 200 64 25
        ##################################################
        # Xp = X.view(X.shape[0],64,-1)
        # Xp = Xp.transpose(1, 2)
        # Xp = Xp.reshape(-1,100,16)
        # Xp = Xp.transpose(1, 2)
        # Xp = Xp.reshape(-1,16,10,10)
        #
        # Xp2 = X2.view(X2.shape[0],64,-1)
        # Xp2 = Xp2.transpose(1, 2)
        # Xp2 = Xp2.reshape(-1,100,16)
        # Xp2 = Xp2.transpose(1, 2)
        # Xp2 = Xp2.reshape(-1,16,10,10)


        Xpa = attention_model(X).view(-1)
        Xpa2 = attention_model(X2).view(-1)
        # print(attention_model.gplength)
        # print(attention_model.spatialattention.conv1.weight)
        Xat = torch.outer(Xpa, Xpa2)  # 5000X5000
        Xat = torch.exp(Xat)
        # Xat = Xat * 1.3
        # Xat = Xat ** attention_model.atts
        # print(attention_model.atts)
        # print(attention_model.gplength)
        Xp = X.view(X.shape[0], X.shape[1], -1)
        Xp2 = X2.view(X2.shape[0], X2.shape[1], -1)
        Xp = Xp.permute(0, 2, 1) #[N,25,64]
        Xp2 = Xp2.permute(0, 2, 1) #[N,25,64]
        Xp = Xp.contiguous().view(-1, X.shape[1])  #
        Xp2 = Xp2.contiguous().view(-1, X2.shape[1])

        bigK = covar_module(Xp, Xp2).cuda()
        bigK = bigK.evaluate()
        bigK = Xat * bigK
        # bigK = bigK * 5

        K = bigK.view(X.shape[0], num_patchs, -1, num_patchs).sum(dim=(1, 3))  # NXN 256x75都不行 676x25可以
    else:
        # Xp = X.reshape(-1, 64, 5, 5)
        # Xp = attention_model(Xp)
        # Xp = X.view(X.shape[0], 64, -1)
        # Xp = Xp.transpose(1, 2)
        # Xp = Xp.reshape(-1, 100, 16)
        # Xp = Xp.transpose(1, 2)
        # Xp = Xp.reshape(-1, 16, 10, 10)

        Xpa = attention_model(X).view(-1)
        Xat = torch.outer(Xpa, Xpa)

        Xat = torch.exp(Xat)
        # Xat = Xat ** attention_model.atts

        Xp = X.view(X.shape[0], X.shape[1], -1)
        Xp = Xp.permute(0, 2, 1) #[N,25,64]
        Xp = Xp.contiguous().reshape(-1, X.shape[1])

        # print("Xp",Xp.shape)

        bigK = covar_module(Xp, Xp).cuda()
        bigK = bigK.evaluate()

        bigK = Xat * bigK
        K = bigK.view(X.shape[0], num_patchs, -1, num_patchs).sum(dim=(1, 3))  # NXN 256x75都不行 676x25可以

    return K / num_patchs ** 2











# import gpytorch
# import torch
#
# covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(100))
# #确定RBFKernel的超参数
# covar_module.base_kernel.lengthscale = 0.6  #94 93
# covar_module.base_kernel.lengthscale = 1.0 #97.0 96.6
# covar_module.base_kernel.lengthscale = 0.9 #97.4 97.1
# covar_module.base_kernel.lengthscale = 0.8 #97.3 97.4
# covar_module.base_kernel.lengthscale = 0.7 #97.2 97.0

# covar_module.base_kernel.lengthscale = 0.65 #97.2 97.1 train 99.6

# covar_module.base_kernel.lengthscale = 0.6 #97.6 97.5  train 99.6
# covar_module.base_kernel.lengthscale = 1 #97.8 97.5  train 98.7
# covar_module.base_kernel.lengthscale = 0.53 #96.1 96.0  train 97.2

# covar_module.base_kernel.lengthscale = 0.5 #97.7 97.5
# covar_module.base_kernel.lengthscale = 0.4 #94.5 94.7
# covar_module.base_kernel.lengthscale = 0.3 #95.3 94.9
# covar_module.base_kernel.lengthscale = 0.1  #95.9 95.9



# def convgp(X, X2=None, global_model=None,attention_model=None):
#     # Xp = get_patches(X)
#     # Xp: N x num_patches x 64
#     # global: N x num_patches(or 64)
#     # 1.Xp+global 成为全局特征
#     # 2.K(Xp)+K(global) 成为全局特征
#     #
#     # 局部不加全局
#     if X2 is not None:
#         # Xp = X.reshape(-1, 8, 20, 10) #200 64 25
#         # Xp2 = X2.reshape(-1, 8, 20, 10)#200 64 25
#         #
#         # Xpa = attention_model(Xp).reshape(-1)
#         # Xpa2 = attention_model(Xp2).reshape(-1)
#         # Xat = torch.outer(Xpa,Xpa2)  #5000X5000
#         Xpa = attention_model(X).reshape(-1)
#         Xpa2 = attention_model(X2).reshape(-1)
#         Xat = torch.outer(Xpa,Xpa2)  #5000X5000
#
#         Xp = X.reshape(-1, 64)
#         Xp2 = X2.reshape(-1, 64)
#         bigK = covar_module(Xp,Xp2).cuda()
#         bigK = bigK.evaluate()
#         bigK = Xat * bigK
#
#         K = bigK.reshape(X.shape[0], 100, -1, 100).sum(dim=(1, 3))  # NXN 256x75都不行 676x25可以
#         torch.cuda.empty_cache()
#     else:
#         # Xp = X.reshape(-1, 64, 5, 5)
#         # Xp = attention_model(Xp)
#         Xp = X.reshape(-1, 64)
#         bigK = covar_module(Xp, Xp).cuda()
#
#         K = bigK.evaluate().reshape(X.shape[0], 100, -1, 100).sum(dim=(1, 3))# NXN 256x75都不行 676x25可以
#         torch.cuda.empty_cache()
#
#     return K / 100 ** 2.0

# Resnet-12   640-5-5
# def convgp(X, X2=None, global_model=None, attention_model=None):
#     # Xp = get_patches(X)
#     # Xp: N x num_patches x 64
#     # global: N x num_patches(or 64)
#     # 1.Xp+global 成为全局特征
#     # 2.K(Xp)+K(global) 成为全局特征
#     #
#     # 局部不加全局
#     if X2 is not None:
#         Xp = X.reshape(-1, 100, 16, 10)  # 200 64 25
#         Xp2 = X2.reshape(-1, 100, 16, 10)  # 200 64 25
#
#         Xpa = attention_model(Xp).reshape(-1)
#         Xpa2 = attention_model(Xp2).reshape(-1)
#         Xat = torch.outer(Xpa, Xpa2)  # 5000X5000
#         # Xat = torch.exp(Xat)
#
#         Xp = Xp.reshape(-1, 100)
#         Xp2 = Xp2.reshape(-1, 100)
#         bigK = covar_module(Xp, Xp2).cuda()
#         bigK = bigK.evaluate()
#         bigK = Xat * bigK
#
#         K = bigK.reshape(X.shape[0], 160, -1, 160).sum(dim=(1, 3))  # NXN 256x75都不行 676x25可以
#         torch.cuda.empty_cache()
#     else:
#         # Xp = X.reshape(-1, 64, 5, 5)
#         # Xp = attention_model(Xp)
#         Xp = X.reshape(-1, 100)
#         bigK = covar_module(Xp, Xp).cuda()
#
#         K = bigK.evaluate().reshape(X.shape[0], 160, -1, 160).sum(dim=(1, 3))  # NXN 256x75都不行 676x25可以
#         torch.cuda.empty_cache()
#
#     return K / 160 ** 2.0
    # #全局
    # #如果X2不是None
    # if X2 is not None:
    #     Xp = X.reshape(-1, 100, 16)  # 200 64 25
    #     Xp2 = X2.reshape(-1, 100, 16)
    #
    #     Xg = X.reshape(-1, 100, 16).mean(dim=2) #200 64 ,xg是全局特征
    #     Xg = global_model(Xg)
    #     Xg = Xg.view(Xg.shape[0],Xg.shape[1],1) #200 64 1
    #     Xg = Xg.expand(Xg.shape[0],Xg.shape[1], 16)#200 64 25
    #
    #     Xg2 = X2.reshape(-1, 100, 16).mean(dim=2)
    #     Xg2 = global_model(Xg2)
    #     Xg2 = Xg2.view(Xg2.shape[0], Xg2.shape[1], 1)
    #     Xg2 = Xg2.expand(Xg2.shape[0], Xg2.shape[1], 16)
    #
    #     Xgl = Xp + Xg
    #     Xgl = Xgl.reshape(-1, 64, 5, 5)
    #
    #     Xgl2 = Xp2 + Xg2
    #     Xgl2 = Xgl2.reshape(-1, 64, 5, 5)
    #
    #     Xgl = attention_model(Xgl)
    #     Xgl2 = attention_model(Xgl2)
    #
    #     Xgl = Xgl.reshape(-1, 64)
    #     Xgl2 = Xgl2.reshape(-1, 64)
    #
    #     # Xg = Xg.reshape(-1,40)
    #     # Xg2 = Xg2.reshape(-1,40)
    #     # Xp = X.reshape(-1, 40)  # 200 64 25
    #     # Xp2 = X2.reshape(-1, 40)
    #     # bigK = covar_module(Xp, Xp2).cuda() + covar_module(Xg, Xg2).cuda() + covar_module(Xp, Xg2).cuda() + covar_module(Xg, Xp2).cuda()  # N * num_patches x N * num_patches
    #     # bigK = covar_module(Xp, Xp2).cuda() + covar_module(Xg, Xg2).cuda()  # N * num_patches x N * num_patches
    #     # K = torch.zeros([X.shape[0],X2.shape[0]])
    #     # for i in range(X.shape[0]):
    #     #     for j in range(X2.shape[0]):
    #     #         K[i][j] = covar_module(Xgl[i], Xgl2[j]).cuda().evaluate().sum()
    #
    #
    #     bigK = covar_module(Xgl, Xgl2).cuda() # N * num_patches x N * num_patches 25x40 75x40
    #     K = bigK.evaluate().reshape(X.shape[0], 25, -1, 25).sum(dim=(1, 3)) # NXN 256x75都不行 676x25可以
    #     torch.cuda.empty_cache()
    #
    # else:
    #     Xp = X.reshape(-1, 100, 16)  # 200 64 25
    #     Xg = X.reshape(-1, 100, 16).mean(dim=2)  # 200 64 ,xg是全局特征
    #     Xg = global_model(Xg)
    #     Xg = Xg.view(Xg.shape[0], Xg.shape[1], 1)  # 200 64 1
    #     Xg = Xg.expand(Xg.shape[0], Xg.shape[1], 16)  # 200 64 25
    #     Xgl = Xp + Xg
    #     Xgl = Xgl.reshape(-1, 64,5, 5)
    #     Xgl = attention_model(Xgl)
    #     Xgl = Xgl.reshape(-1, 64)
    #     bigK = covar_module(Xgl, Xgl).cuda()  # N * num_patches x N * num_patches
    #
    #     K = bigK.evaluate().reshape(X.shape[0], 25, -1, 25).sum(dim=(1, 3))# NXN 256x75都不行 676x25可以
    #     torch.cuda.empty_cache()

    # return K / 25 ** 2.0
        # K = torch.zeros([X.shape[0], X.shape[0]])
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[0]):
        #         K[i][j] = covar_module(Xp[i], Xp[j]).cuda().evaluate().sum()



# Xp = self._get_patches(X)
# Xp = Xp.reshape(-1, self.patch_len)

    #如果X2不是None
    # if X2 is not None:
    #     K = torch.zeros([X.shape[0],X2.shape[0]])
    #     for i in range(X.shape[0]):
    #         for j in range(X2.shape[0]):
    #             K[i][j] = covar_module(Xp[i], Xp2[j]).cuda().evaluate().sum()
    # else:
    #     K = torch.zeros([X.shape[0], X.shape[0]])
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[0]):
    #             K[i][j] = covar_module(Xp[i], Xp[j]).cuda().evaluate().sum()