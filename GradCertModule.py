import copy
import torch
import numpy as np
from torch.nn import functional as F
from typing import Union


#GLOBAL_MODEL_PETURB = 'absolute'
GLOBAL_MODEL_PETURB = 'relative'
GLOBAL_USING_GPU = False
GLOBAL_PRE_FLATTEN = [0]
"""
Training Helpers for Pytorch Lightning
"""
def affine_forward(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    This function uses pytorch to compute upper and lower bounds
    on a matrix multiplication given bounds on the matrix 'x' 
    as given by x_l (lower) and x_u (upper)
    """
    marg = marg/2; b_marg = b_marg/2
    x_mu = (x_u + x_l)/2
    x_r = (x_u - x_l)/2
    W_mu = W
    if(GLOBAL_MODEL_PETURB == 'relative'):
        W_r =  torch.abs(W)*marg
    elif(GLOBAL_MODEL_PETURB == 'absolute'):
        W_r =  torch.ones_like(W)*marg
    b_u =  torch.add(b, b_marg)
    b_l =  torch.subtract(b, b_marg)
    h_mu = torch.matmul(x_mu, W_mu.T)
    x_rad = torch.matmul(x_r, torch.abs(W_mu).T)
    #assert((x_rad >= 0).all())
    W_rad = torch.matmul(torch.abs(x_mu), W_r.T)
    #assert((W_rad >= 0).all())
    Quad = torch.matmul(torch.abs(x_r), torch.abs(W_r).T)
    #assert((Quad >= 0).all())
    h_u = torch.add(torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = torch.add(torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad), b_l)
    return h_l, h_u

def conv2d_forward(W, b, x_l, x_u, marg=0, b_marg=0, stride=1):
    marg = marg/2; b_marg = b_marg/2
    x_mu = (x_u + x_l)/2
    x_r = (x_u - x_l)/2
    #W = torch.Tensor(W)
    W_mu = W
    if(GLOBAL_MODEL_PETURB == 'relative'):
        W_r =  torch.abs(W)*marg
    elif(GLOBAL_MODEL_PETURB == 'absolute'):
        W_r =  torch.ones_like(W)*marg
    # https://discuss.pytorch.org/t/adding-bias-to-convolution-output/82684/5
    b_size = b.shape[0]
    b = torch.reshape(b, (1, b_size, 1, 1))
    b_u =  torch.add(b, b_marg)
    b_l =  torch.subtract(b, b_marg)
    h_mu = torch.nn.functional.conv2d(x_mu, W_mu, stride=stride)
    x_rad = torch.nn.functional.conv2d(x_r, torch.abs(W_mu), stride=stride)
    #assert((x_rad >= 0).all())
    W_rad = torch.nn.functional.conv2d(torch.abs(x_mu), W_r, stride=stride)
    #assert((W_rad >= 0).all())
    Quad = torch.nn.functional.conv2d(torch.abs(x_r), torch.abs(W_r), stride=stride)
    #assert((Quad >= 0).all())
    h_u = torch.add(torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = torch.add(torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad), b_l)
    assert((h_u >= h_l).all())
    return h_l, h_u

def propagate_matmul(W, x_l, x_u, marg=0):
    """
    This function uses pytorch to compute upper and lower bounds
    on a matrix multiplication given bounds on the matrix 'x' 
    as given by x_l (lower) and x_u (upper)
    """
    marg = marg/2 #; b_marg = b_marg/2
    x_mu = (x_u + x_l)/2
    x_r = (x_u - x_l)/2
    #marg = torch.divide(marg, 2)
    #x_mu = torch.divide(torch.add(x_u, x_l), 2)
    #x_r =  torch.divide(torch.subtract(x_u, x_l), 2)
    W_mu = W
    if(GLOBAL_MODEL_PETURB == 'relative'):
        W_r =  torch.abs(W)*marg
    elif(GLOBAL_MODEL_PETURB == 'absolute'):
        W_r =  torch.ones_like(W)*marg
    h_mu = torch.matmul(x_mu, W_mu)
    x_rad = torch.matmul(x_r, torch.abs(W_mu))
    #assert((x_rad >= 0).all())
    W_rad = torch.matmul(torch.abs(x_mu), W_r)
    #assert((W_rad >= 0).all())
    Quad = torch.matmul(torch.abs(x_r), torch.abs(W_r))
    #assert((Quad >= 0).all())
    h_u = torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad)
    h_l = torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad)
    #assert((h_u >= h_l).all())
    return h_l, h_u


def backwards_conv2d(W, x_l, x_u, backconv, marg=0, b_marg=0):
    marg = marg/2; b_marg = b_marg/2
    x_mu = (x_u + x_l)/2
    x_r = (x_u - x_l)/2
    W_mu = W
    if(GLOBAL_MODEL_PETURB == 'relative'):
        W_r =  torch.abs(W)*marg
    elif(GLOBAL_MODEL_PETURB == 'absolute'):
        W_r =  torch.ones_like(W)*marg    
    h_mu = backconv(x_mu) 
    backconv.weight = torch.nn.Parameter(torch.abs(W_mu))
    x_rad = backconv(x_r)
    #assert((x_rad >= 0.0).all())
    backconv.weight = torch.nn.Parameter(W_r)
    # Needs to be greater than zero everywhere?
    W_rad = backconv(torch.abs(x_mu))
    #assert((W_rad >= 0.0).all())
    backconv.weight = torch.nn.Parameter(torch.abs(W_r))
    Quad = backconv(torch.abs(x_r))
    h_u = torch.add(torch.add(torch.add(h_mu, x_rad), W_rad), Quad)
    h_l = torch.subtract(torch.subtract(torch.subtract(h_mu, x_rad), W_rad), Quad)
    return h_l, h_u


def IntervalBoundForward(model, weights, inp, eps, gam):
    h_l = inp-eps; h_u = inp+eps
    assert((h_l <= h_u).all())
    # We need to determine if it is a CNN for offset reasons :) 
    FCN_FLAG = True
    for i in range(len(model.layers)):
        if("CONV" in model.layers[i].upper()):
            FCN_FLAG = False
    FCN_FLAG = int(FCN_FLAG)
    inter_l = [h_l]; inter_u = [h_u]
    inds_l = []; inds_u = []
    layers = int(len(weights)/2); 
    pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    first_conv = True
    offset = 0 
    for i in range(len(model.layers)):
        if("LINEAR" in model.layers[i].upper()):
            w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
            h_l, h_u = affine_forward(w, b, h_l, h_u, marg=gam, b_marg=gam)
            #assert((h_l <= h_u).all())
            if(i < layers-FCN_FLAG):
                h_l = model.activations[0](h_l) 
                h_u = model.activations[0](h_u)
            inter_l.append(h_l); inter_u.append(h_u)  
        elif("CONV" in model.layers[i].upper()):
            stride = int(model.layers[i].upper()[4])
            w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
            h_l, h_u = conv2d_forward(w, b, h_l, h_u, marg=gam, b_marg=gam, stride=stride)
            #assert((h_l <= h_u).all())
            if(i < layers-1):
                h_l = model.activations[0](h_l) 
                h_u = model.activations[0](h_u)
            inter_l.append(h_l); inter_u.append(h_u)  
        elif("POOL" in model.layers[i].upper()):
            h_l, ind_l = pool(h_l)
            h_u, ind_u = pool(h_u) 
            inds_l.append(ind_l); inds_u.append(ind_u)
            inter_l.append(h_l); inter_u.append(h_u)
            offset += 1
        elif("FLAT" in model.layers[i].upper()):
            h_u = torch.flatten(h_u, 1)
            h_l = torch.flatten(h_l, 1)
            inter_l.append(h_l); inter_u.append(h_u)
            offset += 1
        #print("Output: ", h_l)
    inter_l.pop(); inter_u.pop() # This is here because we compute the last layer partial ourselves
    return h_l, h_u, inter_l, inter_u, inds_l, inds_u


def ReLuBackwardsBounds(inter_l, inter_u):
    min_deriv = torch.sign(inter_l)
    max_deriv = torch.sign(inter_u)  
    return min_deriv, max_deriv

def TanhBackwardsBounds(inter_l, inter_u):
    derivative_one = torch.sign(inter_l) != torch.sign(inter_u) 
    min_abs = torch.minimum(torch.abs(inter_l), torch.abs(inter_u))
    max_abs = torch.maximum(torch.abs(inter_l), torch.abs(inter_u))
    max_deriv = 1-(torch.tanh(min_abs)**2)
    min_deriv = 1-(torch.tanh(max_abs)**2)
    max_deriv = torch.maximum(derivative_one, max_deriv)
    return min_deriv, max_deriv

def SigmoidBackwardsBounds(inter_l, inter_u):
    #e^x/(1 + e^x)^2
    derivative_q = 0.25*(torch.sign(inter_l) != torch.sign(inter_u))
    min_abs = torch.minimum(torch.abs(inter_l), torch.abs(inter_u))
    max_abs = torch.maximum(torch.abs(inter_l), torch.abs(inter_u))
    max_deriv = torch.exp(min_abs)/((1 + torch.exp(min_abs))**2)
    min_deriv = torch.exp(max_abs)/((1 + torch.exp(max_abs))**2)
    max_deriv = torch.maximum(derivative_q, max_deriv)
    return min_deriv, max_deriv

def ELuBackwardsBounds(inter_l, inter_u):
    assert(False, "Not implimented for ELU yet!")
    return min_inter, max_inter
            
def IntervalBoundBackward(model, y_l, y_u, y_t, weights, inter_l, inter_u, inds_l, inds_u, gam):
    unpool = torch.nn.MaxUnpool2d(2, stride=2)
    dL_max = torch.maximum((y_l*y_t - y_t), (y_u*y_t - y_t))
    dL_min = torch.minimum((y_l*y_t - y_t), (y_u*y_t - y_t))
    #print("logit grads: ", dL_max, dL_min)
    assert((dL_min <= dL_max).all())
    #print(dL_max, dL_min)
    layers = int(len(weights)/2)
    convs = 0; weight_layers = 0; offset = 0
    for i in range(len(model.layers)-1, -1, -1):
        if("LINEAR" in model.layers[i].upper()):
            weight_layers += 1
            # Later this will need to be replaced with min and max activation derivative
            #assert((inter_l[(weight_layers*-1)-offset] <=  inter_u[(weight_layers*-1)-offset]).all())
            min_inter = torch.sign(inter_l[(weight_layers*-1)-offset])
            max_inter = torch.sign(inter_u[(weight_layers*-1)-offset]) 
            dL_dz_min, dL_dz_max = propagate_matmul(weights[(2*(weight_layers*-1))], dL_min, dL_max, marg=gam)
            #assert((dL_dz_min <= dL_dz_max).all())
        elif("CONV" in model.layers[i].upper()):
            stride = int(model.layers[i].upper()[4])
            weight_layers += 1;
            # Later this will need to be replaced with min and max activation derivative
            #assert((inter_l[(weight_layers*-1)-offset] <=  inter_u[(weight_layers*-1)-offset]).all())
            min_inter = torch.sign(inter_l[(weight_layers*-1)-offset])
            max_inter = torch.sign(inter_u[(weight_layers*-1)-offset]) 
            w = weights[(2*(weight_layers*-1))]
            # Grab shapes to define the deconvolution opperation
            filt_size = weights[(2*(weight_layers*-1))].shape[-1]
            num_filts = weights[(2*(weight_layers*-1))].shape[1]
            deconv = torch.nn.ConvTranspose2d(num_filts, w.shape[1], filt_size, stride=stride)
            deconv.weight = torch.nn.Parameter(w)
            if(GLOBAL_USING_GPU):
                deconv.bias = torch.nn.Parameter(torch.zeros_like(deconv.bias).cuda())
            else:
                deconv.bias = torch.nn.Parameter(torch.zeros_like(deconv.bias))
            # Compute the desired input shape for convolution
            d = list(inter_l[(weight_layers*-1)].shape[1:])
            s = [-1] + d 
            dL_min, dL_max = torch.reshape(dL_min, s), torch.reshape(dL_max, s)
            if( "POOL" in model.layers[i+1].upper()):
                print("Warning: Using Unpooling leads to invalid bounds due to improper inversion.")
                offset += 1
                # If Pool, We grabbed the wrong inter stage so we correct that here:
                min_inter = torch.sign(inter_l[(weight_layers*-1)-offset])
                max_inter = torch.sign(inter_u[(weight_layers*-1)-offset]) 
                convs += 1
                # Need to handle an off by one error in unpooling odd shapes
                shape_1 = list(dL_min.shape); 
                shape_1[0] = -1; shape_1[-1]*=2; shape_1[-2]*=2
                shape_2 = list(dL_min.shape); 
                shape_2[0] = -1; shape_2[-1] = (shape_2[-1] * 2)+1; shape_2[-2] = (shape_2[-2] * 2)+1
                try:
                    dL_min_1 = unpool(dL_min, inds_l[-convs], output_size=shape_1)
                    dL_max_1 = unpool(dL_max, inds_u[-convs], output_size=shape_1)
                    dL_min = dL_min_1; dL_max = dL_max_1
                except:
                    dL_min_2 = unpool(dL_min, inds_l[-convs], output_size=shape_2)
                    dL_max_2 = unpool(dL_max, inds_u[-convs], output_size=shape_2)
                    dL_min = dL_min_2; dL_max = dL_max_2
            dL_dz_min, dL_dz_max = backwards_conv2d(w, dL_min, dL_max, deconv, marg=gam)
            #print("diff between conv max and min: ", torch.mean(dL_dz_min - dL_dz_max))
            #assert((dL_dz_min <= dL_dz_max).all())
        elif("FLAT" in model.layers[i].upper()):
            offset += 1
            continue
        if(i > 1):
            # I guess it is only a few more operations to do the propagation through the elementwise 
            #    multiplication so lets do it just to be sure :) 
            intermed_a, intermed_b = torch.mul(dL_dz_min, min_inter), torch.mul(dL_dz_max, max_inter)
            intermed_c, intermed_d = torch.mul(dL_dz_max, min_inter), torch.mul(dL_dz_min, max_inter)
            a_min = torch.minimum(intermed_a, intermed_b)
            b_min = torch.minimum(intermed_c, intermed_d)
            dL_min = torch.minimum(a_min, b_min)
            a_max = torch.maximum(intermed_a, intermed_b)
            b_max = torch.maximum(intermed_c, intermed_d)
            dL_max = torch.maximum(a_max, b_max)
        else:
            dL_min = torch.minimum(dL_dz_min, dL_dz_max)
            dL_max = torch.maximum(dL_dz_min, dL_dz_max)
        #print(model.layers[i].upper())
        #print("layer %s: "%(i), dL_max, dL_min)
        #assert((dL_min <= dL_max).all())
    return dL_min.T, dL_max.T #dL_dz_min.T, dL_dz_max.T


def PostProcessBounds(lower, upper):
    # Gives us zeros where the sign changes and large value otherwise
    min_zero = 10000*(1-((torch.sign(lower) != torch.sign(upper)).long()))
    min_abs = torch.minimum(torch.abs(lower), torch.abs(upper))
    max_abs = torch.maximum(torch.abs(lower), torch.abs(upper))
    min_abs = torch.minimum(min_abs, min_zero)
    return min_abs, max_abs

def GradCertRegularizer(model, inp, lab, eps, gam, nclasses=10):
    lab = lab.to(torch.int64)
    #print("GOT LABEL: ", lab, type(lab))
    weights = [t for t in model.parameters()]
    logit_l, logit_u, inter_l, inter_u, inds_l, inds_u = IntervalBoundForward(model, weights, inp, eps, gam)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses) 
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = torch.nn.Softmax(dim=1)(worst_case)
    y_u = torch.nn.Softmax(dim=1)(best_case)
    min_grad, max_grad = IntervalBoundBackward(model, y_l, y_u, v1, weights, inter_l, inter_u, inds_l, inds_u, gam)
    return torch.mean(max_grad - min_grad) * len(inp)

def RobustnessRegularizer(model, inp, lab, eps, gam, nclasses=10):
    weights = [t for t in model.parameters()]
    logit_l, logit_u, inter_l, inter_u, inds_l, inds_u = IntervalBoundForward(model, weights, inp, eps, gam)
    v1 = torch.nn.functional.one_hot(lab.long(), num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab.long(), num_classes=nclasses) 
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    return  F.cross_entropy(worst_case, lab)

def RobustnessBounds(model, inp, lab, eps, gam, nclasses=10):
    weights = [t for t in model.parameters()]
    logit_l, logit_u, inter_l, inter_u, inds_l, inds_u = IntervalBoundForward(model, weights, inp, eps, gam)
    #print(logit_l)
    #print(logit_u)
    v1 = torch.nn.functional.one_hot(lab.long(), num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab.long(), num_classes=nclasses) 
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = worst_case #torch.nn.Softmax(dim=1)(worst_case)
    y_u = best_case #torch.nn.Softmax(dim=1)(best_case)
    return y_l, y_u


def HessianRegularizer(model, inp, lab):
    inp.requires_grad = True

    # hessian regularization
    lab_hat = model(inp)
    criterion_fn = lambda output, y: output[range(len(lab)), lab].sum()
    v = torch.randn_like(inp)
    try:
        g = criterion_fn(lab_hat, lab)
    except:
        criterion_fn = lambda output, y: output[:,y].sum()
        def criterion_fn(lab_hat, lab):
            s = 0
            for i in range(len(lab_hat)):
                #s += torch.squeeze(lab_hat[i])[lab[i].long()]
                s += torch.abs(torch.max(lab_hat[i]))
            #print("s val ", s)
            return s
        g = criterion_fn(lab_hat, lab)
        
    # Get grad
    grad1 = torch.autograd.grad(outputs=g, inputs=inp, create_graph=True, retain_graph=True)[0]
    try:
        dot_vg_vec = torch.einsum("nchw,nchw->n", v, grad1)
    except: 
        dot_vg_vec = torch.einsum("bd,bd->b", v, grad1)
    # Get grad grad
    grad2 = torch.autograd.grad(outputs=dot_vg_vec.sum(), inputs=inp, create_graph=True, retain_graph=True)[0]
    try:
        fn_sq = torch.einsum("nchw,nchw->n", grad2, grad2)
    except:
        fn_sq = torch.einsum("bd,bd->b", grad2, grad2)
    # Get loss
    reg_loss = torch.abs(fn_sq).mean()
    #print(reg_loss)
    return reg_loss


def L2Regularizer(model, inp, lab, std, eps=4):
    inp.requires_grad = True

    # Get grad on inp
    lab_hat = model(inp)
    criterion_fn = lambda output, y: output[range(len(y)), y].sum()
    try:
        labc_hat = criterion_fn(lab_hat, lab)
    except:
        def criterion_fn(lab_hat, lab):
            s = 0
            for i in range(len(lab_hat)):
                #s += torch.squeeze(lab_hat[i])[lab[i].long()]
                s += torch.abs(torch.max(lab_hat[i]))
            #print("s val ", s)
            return s
        labc_hat = criterion_fn(lab_hat, lab)
    grad = torch.autograd.grad(outputs=labc_hat, inputs=inp, create_graph=True, retain_graph=True)[0]

    # Get grad on inp+noise
    eps_into_norm = model.eps # / (255 * std)
    z = eps_into_norm * (2 * torch.rand_like(inp) - 1)
    inp_r = (inp + z).clone().detach().requires_grad_()
    lab_hat_r = model(inp_r)
    labc_hat_r = criterion_fn(lab_hat_r, lab)
    grad_r = torch.autograd.grad(outputs=labc_hat_r, inputs=inp_r, create_graph=True, retain_graph=True)[0]

    # Get L2distance
    reg_loss = (grad - grad_r).flatten(start_dim=1).norm(dim=1).square().mean()
    return reg_loss


def PGDRegularizer(model, inp, lab, eps=0.1, iters=10):
    adv_inp = copy.deepcopy(inp)
    adv_inp.requires_grad = True
    loss = torch.nn.CrossEntropyLoss()
    for i in range(iters) :
        adv_inp.requires_grad = True
        outputs = model(adv_inp)
        model.zero_grad()
        cost = loss(outputs, lab)#.to(device)
        cost.backward()
        adv_inp = adv_inp + (eps/iters)*adv_inp.grad.sign()
        adv_inp = torch.clamp(adv_inp, min=0, max=1).detach_()
    # Compute loss on these 
    outputs = model(adv_inp)
    reg_loss = loss(outputs, lab)#.to(device)
    return reg_loss


def L2AdvRegularizer(model, inp, lab, eps=0.1, iters=10):
    #adv_inp = copy.deepcopy(inp)
    #adv_inp.requires_grad = True
    inp.requires_grad = True
    loss = torch.nn.L1Loss()
    
    # Get grad on inp
    def get_grad(model, inp, lab):
        inp.requires_grad = True
        lab_hat = model(inp)
        criterion_fn = lambda output, y: output[range(len(y)), y].sum()
        try:
            labc_hat = criterion_fn(lab_hat, lab)
        except:
            def criterion_fn(lab_hat, lab):
                s = 0
                for i in range(len(lab_hat)):
                    s += torch.abs(torch.max(lab_hat[i]))
                return s
            labc_hat = criterion_fn(lab_hat, lab)
        grad = torch.autograd.grad(outputs=labc_hat, inputs=inp, create_graph=True, retain_graph=True)[0]
        return grad
    
    grad_orig = get_grad(model, inp, lab)
    eps_into_norm = model.eps # / (255 * std)
    z = eps_into_norm * (2 * torch.rand_like(inp) - 1)
    adv_inp = (inp + z).clone().detach().requires_grad_()
    #adv_inp.requires_grad = True
    for i in range(iters) :
        adv_inp.requires_grad = True
        grad = get_grad(model, adv_inp, lab)
        model.zero_grad()
        # We want to maximize the MSE by perturbing the input
        cost = -1 * loss(grad, grad_orig)
        cost.backward(retain_graph=True)
        adv_inp = adv_inp + (eps/iters)*adv_inp.grad.sign()
        adv_inp = torch.clamp(adv_inp, min=0, max=1).detach_()
    # Compute loss on these 
    grad = get_grad(model, adv_inp, lab)
    # Return the adv worst-case loss to minimize:
    reg_loss = loss(grad, grad_orig)
    return reg_loss




"""
Evaluation Functions and Helpers:
"""
def GradCertBounds(model, inp, lab, eps, gam, nclasses=10):
    weights = [t for t in model.parameters()]
    logit_l, logit_u, inter_l, inter_u, inds_l, inds_u = IntervalBoundForward(model, weights, inp, eps, gam)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses) 
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = worst_case #torch.nn.Softmax(dim=1)(worst_case) #- 0.000005 
    y_u = best_case #torch.nn.Softmax(dim=1)(best_case)  #+ 0.000005
    #print("Worst: ", y_l)
    #print("Best : ", y_u)
    min_grad, max_grad = IntervalBoundBackward(model, y_l, y_u, v1, weights, inter_l, inter_u, inds_l, inds_u, gam)
    #print(min_grad)
    #print(max_grad)
    return min_grad, max_grad


"""
Evaluation Helpers
"""
def InputGrad(model, data, target, nclasses=10):
    # Small deviations in torch computations
    # versus ours can lead to some invalid comparisons
    a, b = GradCertBounds(model, data, target, 0.0, 0.0, nclasses=nclasses)
    return a

def _INGRAD(model, data, target, nclasses=10):
    device = torch.device("cpu")
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    try:
        output = torch.nn.Softmax(dim=1)(output)
    except:
        output = torch.nn.Softmax(dim=0)(output)
        output = output[None, :]
    #init_pred = output.max(1, keepdim=True)[1] 
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    return data_grad

import skimage
from tqdm import trange 

def run_mnist_attack(model, x, target = 1, epsilon=0.2, iterations=250, lr=0.025, shape=(28,28), label_reg=0.05):
    model.inputfooling_ON()
    y, cls = model.classify(torch.Tensor(x[None, :]))
    #y, cls = model.classify(torch.Tensor(x))
    noise = torch.rand(x.shape)*0.1
    x_adv = torch.Tensor(x + noise).clone().detach()
    
    def get_gradient(model, x, desired_index):
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 30
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)

        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap
    
    # produce expls
    x.requires_grad = True
    org_expl = get_gradient(model, x, cls)
    org_expl = org_expl.detach().cpu()
    x = x.detach()
    minimize = 1
    # Generate target
    if(type(target) == int):
        target_locations = [[6,6], [14, 6], [22,6], [6, 22], [14, 22], [22,22]]
        targ_i, targ_j = target_locations[target]
        targex = np.asarray(x.detach().numpy()).reshape(shape) * 0.0
        targex[targ_i, targ_j] = 10
        sigma = 3.0
        targex = skimage.filters.gaussian(
            targex, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
        targex = (targex-np.min(targex))/(np.max(targex)-np.min(targex))
        targex *= 10 * np.max(org_expl.detach().numpy())
    elif(type(target) == list):
        targ_i, targ_j = target
        targex = np.asarray(x.detach().numpy()).reshape(shape) * 0.0
        targex[targ_i-4:targ_i+4,targ_j-4:targ_j+4] = 1
    else:
        x_temp = torch.Tensor(x_adv[None, :])
        x_temp.requires_grad = True
        targex = get_gradient(model, x_temp, cls)
        minimize = -1
        
    target_expl = torch.Tensor(targex)
    target_expl = target_expl.detach()
    # Generate adversarial attack
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True

    optimizer = torch.optim.Adam([x_adv], lr=lr)
    value = 10
    t = trange(iterations, desc="Loss: %s"%(value))
    for i in t:
    #for i in range(iterations):
        optimizer.zero_grad()

        y_adv, cls = model.classify(x_adv)
        grad_adv = get_gradient(model, x_adv, cls)
        #print(grad_adv.shape, target_expl.shape)
        
        grad_adv = torch.squeeze(grad_adv)
        target_expl = torch.squeeze(target_expl)
        #loss_expl = F.mse_loss(grad_adv[targ_i-4:targ_i+4,targ_j-4:targ_j+4], 
        #                       target_expl[targ_i-4:targ_i+4,targ_j-4:targ_j+4])
        loss_expl = F.mse_loss(grad_adv, target_expl)
        
        loss_output = F.mse_loss(y_adv, y.detach())
 
        total_loss = minimize*(0.5 * loss_expl) + (label_reg * loss_output)

        total_loss.backward()
        optimizer.step()
        #print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl.item(), loss_output.item()))         
        x_adv.data = torch.clamp(x_adv.data, x - epsilon, x + epsilon)
        x_adv.data = torch.clamp(x_adv.data, 0, 1)
        value = total_loss.item()
        t.set_description("Loss: %s"%(value))
        t.refresh() # to show immediately the update
    
    success = False
    adv_result = grad_adv.detach().numpy()
    inds = np.argmax(adv_result)
    max_i = int(inds/28); max_j = inds%28
    if(type(target) == int):
        if(max_i > targ_i-3 and max_i < targ_i+3 and max_j > targ_j-3 and max_j < targ_j+3):
            success = True
    elif(type(target) == list):
        if(max_i > targ_i-3 and max_i < targ_i+3 and max_j > targ_j-3 and max_j < targ_j+3):
            success = True
    else:
        if(max_i > 23 or max_j > 23 or max_i < 5 or max_j < 5):
            success = True
        
    return success, x_adv, grad_adv, target_expl, total_loss.data



def run_model_attack(model, x, target = 1, gamma=0.1, iterations=250, lr=0.025, shape=(28,28), label_reg=0.05):
    model.inputfooling_ON()
    y, cls = model.classify(torch.Tensor(x[None, :]))
    noise = torch.rand(x.shape)*0.1
    x_adv = torch.Tensor(x).clone().detach()
    
    def get_gradient(model, x, desired_index):
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 30
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)

        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap
    
    # produce expls
    x.requires_grad = True
    org_expl = get_gradient(model, x, cls)
    org_expl = org_expl.detach().cpu()
    x = x.detach()
    minimize = 1
    
    # Generate target
    if(type(target) == int):
        target_locations = [[6,6], [14, 6], [22,6], [6, 22], [14, 22], [22,22]]
        targ_i, targ_j = target_locations[target]
        targex = np.asarray(x.detach().numpy()).reshape(shape) * 0.0
        targex[targ_i, targ_j] = 10
        sigma = 3.0
        targex = skimage.filters.gaussian(
            targex, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
        targex = (targex-np.min(targex))/(np.max(targex)-np.min(targex))
        targex *= 10 * np.max(org_expl.detach().numpy())
        
    elif(type(target) == list):
        targ_i, targ_j = target
        targex = np.asarray(x.detach().numpy()).reshape(shape) * 0.0
        #targex += 1
        targex[targ_i-4:targ_i+4,targ_j-4:targ_j+4] = 1
        
    else:
        x_temp = torch.Tensor(x_adv[None, :])
        x_temp.requires_grad = True
        targex = get_gradient(model, x_temp, cls)
        minimize = -1
        
    target_expl = torch.Tensor(targex)
    target_expl = target_expl.detach()
    
    # Generate adversarial attack
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True

    weights = [t for t in model.parameters()]
    weights_upper = [t + (gamma*t) for t in model.parameters()]
    weights_lower = [t - (gamma*t) for t in model.parameters()]
    
    optimizer = torch.optim.Adam(weights, lr=lr)
    #value = 10; t = trange(iterations, desc="Loss: %s"%(value))
    #for i in t:
    for i in range(iterations):
        optimizer.zero_grad()

        y_adv, cls = model.classify(x_adv)
        grad_adv = get_gradient(model, x_adv, cls)
        #print(grad_adv.shape, target_expl.shape)
        loss_expl = F.mse_loss(grad_adv, target_expl)
        loss_output = F.mse_loss(y_adv, y.detach())
 
        total_loss = minimize*(0.5 * loss_expl) + (label_reg * loss_output)

        total_loss.backward()
        optimizer.step()       
        x_adv.data = torch.clamp(x_adv.data, x, x)
        for i in range(len(weights)):
            weights[i].data = torch.clamp(weights[i].data, weights_lower[i], weights_upper[i])
        
        value = total_loss.item()
        #t.set_description("Loss: %s"%(value))
        #t.refresh() # to show immediately the update
    print(value)
    success = False
    adv_result = grad_adv.detach().numpy()
    inds = np.argmax(adv_result)
    max_i = int(inds/28); max_j = inds%28
    if(type(target) == int):
        if(max_i > targ_i-3 and max_i < targ_i+3 and max_j > targ_j-3 and max_j < targ_j+3):
            success = True
    elif(type(target) == list):
        if(max_i > targ_i-3 and max_i < targ_i+3 and max_j > targ_j-3 and max_j < targ_j+3):
            success = True
    else:
        if(max_i > 23 or max_j > 23 or max_i < 5 or max_j < 5):
            success = True
       
    
    return success, weights, grad_adv, target_expl



import copy
import skimage
from tqdm import trange 
def run_tabular_attack(model, x, target = 1, epsilon=0.2, iterations=250, lr=0.02, idx=4):
    model.inputfooling_ON()
    y = model(torch.Tensor(x[None, :]))
    cls = torch.argmax(y)
    noise = torch.rand(x.shape)*0.0
    x_adv = torch.Tensor(x + noise).clone().detach()
    
    def get_gradient(model, x, desired_index):
        num_summands = 5
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap

    # produce expls
    x.requires_grad = True
    org_expl = get_gradient(model, x, cls)
    org_expl = org_expl.detach().cpu()
    x = x.detach()
    
    # Generate target
    targ = target
    target_expl = org_expl
    ma = max(torch.abs(org_expl))
    target_expl *= 0
    target_expl[targ] = ma*5
    
    top_idx = np.squeeze(np.argsort(target_expl.detach().numpy()))
    top_idx = list(reversed(top_idx))
    #print("expl: ", top_idx[0:8], target)
    
    # Generate adversarial attack
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True

    optimizer = torch.optim.SGD([x_adv], lr=lr)
    value = 10
    #t = trange(iterations, desc="Loss: %s"%(value))
    t = range(iterations)
    for i in t:
        optimizer.zero_grad()

        #y_adv, cls = model.classify(x_adv)
        y_adv = model(x_adv)
        #print(x_adv[0], y_adv)
        grad_adv = get_gradient(model, x_adv, cls)
        #print(grad_adv[0][0])
        if(target == -1):
            loss_expl = -1*F.mse_loss(grad_adv, org_expl)
        else:
            loss_expl = F.mse_loss(grad_adv, target_expl)
        loss_output = F.mse_loss(y_adv, y.detach())
        total_loss = (1*loss_expl) - 0.05 * loss_output
        
        total_loss.backward()
        optimizer.step()

        x_adv.data = torch.clamp(x_adv.data, x - epsilon, x + epsilon)
        value = total_loss.item()
        #t.set_description("Loss: %s"%(value))
        #t.refresh() # to show immediately the update
    #print(value)
    grad_adv = get_gradient(model, x_adv, cls)
    #success = False
    #a = np.abs(grad_adv.detach().numpy())
    a = grad_adv.detach().numpy()
    top_idx = np.squeeze(np.argsort(a))
    top_idx = list(reversed(top_idx))
    #print(top_idx[0:idx], target)
    #success = bool(set(top_idx[0:idx]) & set(target))
    if(target == -1):
        diff = np.abs(grad_adv.detach().numpy() - org_expl.detach().numpy())
        success = np.mean(diff)
    else:
        success = bool(set(top_idx[0:idx]) & set(target))
    return success, x_adv, grad_adv

def run_tabular_model_attack(model, x, target = 1, gamma=0.2, iterations=250, lr=0.025, idx=5, tau=0.0):
    model.inputfooling_ON()
    y = model(torch.Tensor(x[None, :]))
    cls = torch.argmax(y)
    
    def get_gradient(model, x, desired_index):
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 100
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)
        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap

    # produce expls
    x.requires_grad = True
    org_expl = get_gradient(model, x, cls)
    org_expl = org_expl.detach().cpu()
    x = x.detach()
    x_adv = torch.Tensor(x).clone().detach()
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True
    
    # Generate target
    targ = target
    target_expl = org_expl * 0.0
    ma = max(torch.abs(org_expl))
    #target_expl /= 5
    #target_expl[targ] = ma*3
    
    target_expl *= 0
    target_expl[targ] = 1
    
    
    # Generate adversarial attack
    for t in model.parameters():
        t.requires_grad = True
    weights = [t for t in model.parameters()]
    weights_upper = [t + (gamma*torch.abs(t)) for t in model.parameters()]
    weights_lower = [t - (gamma*torch.abs(t)) for t in model.parameters()]

    #optimizer = torch.optim.Adam(weights, lr=lr)
    value = 10
    #tracker = trange(iterations, desc="Loss: %s"%(value))
    tracker = range(iterations)
    for i in tracker:
        #optimizer.zero_grad()
        y_adv, cls = model.classify(x_adv)
        grad_adv = get_gradient(model, x_adv, cls)
        #print(grad_adv, value)
        if(target == -1):
            loss_expl = -1*F.mse_loss(grad_adv, org_expl)
        else:
            loss_expl = F.mse_loss(grad_adv, target_expl)
        loss_output = F.mse_loss(y_adv, y.detach())
        total_loss = loss_expl  + (0.05 * loss_output)  - (tau*torch.sum(grad_adv))

        total_loss.backward()
        #optimizer.step()       
        value = total_loss.item()
        
        ind = 0
        for name, param in model.named_parameters():
            values = weights[ind] - (lr * torch.sign(weights[ind].grad)) 
            #values = torch.rand(weights[ind].shape)
            values = torch.clamp(values, weights_lower[ind], weights_upper[ind])
            param.data = values
            ind += 1
            
        x_adv.data = torch.clamp(x_adv.data, x, x)
        #for t in model.parameters():
        #    t.requires_grad = True
         
        #print(weights[0][0][0], weights_lower[0][0][0], weights_upper[0][0][0])
        #print("G: ", weights[0].grad[0][0])
        
        value = total_loss.item()
        #tracker.set_description("Loss: %s"%(value))
        #tracker.refresh() 
    
    #success = False
    a = np.abs(grad_adv.detach().numpy())
    top_idx = np.squeeze(np.argsort(a))
    top_idx = list(reversed(top_idx))
    #success = bool(set(top_idx[0:idx]) & set(target))
    if(target == -1):
        diff = np.abs(grad_adv.detach().numpy() - org_expl.detach().numpy())
        success = np.mean(diff)
    else:
        success = bool(set(top_idx[0:idx]) & set(target))
    return success, x_adv, grad_adv


def run_tabular_model_attack_FGSM(model, x, target = 1, gamma=0.2, iterations=250, lr=0.025, idx=5, tau=0.0):
    model.inputfooling_ON()
    y = model(torch.Tensor(x[None, :]))
    cls = torch.argmax(y)
    def get_gradient(model, x, desired_index):
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 100
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)
        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap
    x.requires_grad = True
    org_expl = get_gradient(model, x, cls)
    org_expl = org_expl.detach().cpu()
    x = x.detach()
    x_adv = torch.Tensor(x).clone().detach()
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True
    
    # Generate target
    targ = target
    target_expl = org_expl * 0.0
    ma = max(torch.abs(org_expl))
    target_expl /= 5
    target_expl[targ] = ma*3
    #target_expl *= 0
    #target_expl[targ] = 1
    
    
    # Generate adversarial attack
    for t in model.parameters():
        t.requires_grad = True
    weights = [t for t in model.parameters()]
    weights_upper = [t + (gamma*torch.abs(t)) for t in model.parameters()]
    weights_lower = [t - (gamma*torch.abs(t)) for t in model.parameters()]

    y_adv, cls = model.classify(x_adv)
    grad_adv = get_gradient(model, x_adv, cls)
    
    loss_expl = F.mse_loss(grad_adv, target_expl)
    loss_output = F.mse_loss(y_adv, y.detach())
    total_loss = loss_expl  + (0.05 * loss_output)  - (tau*torch.sum(grad_adv))
    total_loss.backward()
        
    ind = 0
    for name, param in model.named_parameters():
        values = weights[ind] - ((weights_upper[ind] - weights_lower[ind]) * torch.sign(weights[ind].grad)) 
        #values = torch.rand(weights[ind].shape)
        values = torch.clamp(values, weights_lower[ind], weights_upper[ind])
        param.data = values
        ind += 1
            
    grad_adv = get_gradient(model, x_adv, cls)
    a = np.abs(grad_adv.detach().numpy())
    top_idx = np.squeeze(np.argsort(a))
    top_idx = list(reversed(top_idx))
    success = bool(set(top_idx[0:idx]) & set(target))
    
    return success, org_expl, grad_adv


def run_med_attack(model, x, target = 1, epsilon=0.2, iterations=250, lr=0.025, shape=(28,28), label_reg=0.05):
    model.inputfooling_ON()
    y, cls = model.classify(torch.Tensor(x[None, :]))
    noise = torch.rand(x.shape)*0.0
    x_adv = torch.Tensor(torch.add(x, noise)).clone().detach()
    
    def get_gradient(model, x, desired_index):
        # calculate the integrand in one batch
        # we use DataParallel mode of model to fit the batch in memory of (multiple) gpu(s)
        num_summands = 30
        prefactors = x.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        parallel_model = torch.nn.DataParallel(model)
        y = parallel_model(prefactors.view(num_summands, 1, 1, 1) * x)

        # we sum the result and then take the derivative (instead of summing derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n
        y = torch.nn.functional.softmax(y, 1)[:, int(desired_index)]
        y = (1 / num_summands) * torch.sum(y / prefactors, dim=0)
        heatmap = torch.autograd.grad(y, x, create_graph=True)[0]
        return heatmap
    
    # produce expls
    x.requires_grad = True
    x = x.detach()

    if(type(target) == list):
        targ_i, targ_j = target
        targex = np.asarray(x.detach().numpy()).reshape(shape) * 0.0
        targex[targ_i-4:targ_i+4,targ_j-4:targ_j+4] = 1
        minimize = 1
        
    target_expl = torch.Tensor(targex)
    target_expl = target_expl.detach()
    
    # Generate adversarial attack
    x_adv = torch.Tensor(x_adv[None, :])
    x_adv.requires_grad = True
    
    y_adv, cls = model.classify(x_adv)
    grad_adv = get_gradient(model, x_adv, cls)
    
    loss_expl = F.mse_loss(grad_adv, target_expl)
    loss_output = F.mse_loss(y_adv, y.detach())
    total_loss = loss_expl  + (0.05 * loss_output) # - (label_reg*torch.sum(grad_adv))
    total_loss.backward()
    
    x_adv = x_adv - (epsilon * torch.sign(x_adv.grad))
            
    grad_adv = get_gradient(model, x_adv, cls)
       
    success = False
    
    return success, x_adv, grad_adv, target_expl, total_loss.data





#==================== UNUSED CODE BLOW, DELETED AFTER NEXT COMMIT =========================



"""
Grad Attack Methods - Testing with Attacks rather than intervals
"""
def GradRandMaximizer(model, inp, lab, eps, iterations=1, transpose_grad=True):
    # Run attack for iterations to maximize the model gradient
    max_grad = np.squeeze(copy.deepcopy(inp.detach().numpy()) - 10000000.0)
    orig_inp = copy.deepcopy(inp.detach().numpy())
    ones = np.ones(orig_inp.shape)
    zeros = np.zeros(orig_inp.shape)
    for i in range(iterations):
        adv = orig_inp + np.random.uniform(zeros, ones*eps)
        gra = InputGrad(model, torch.Tensor(adv), lab)
        gra = gra.detach().numpy()
        if(transpose_grad):
            gra = gra.T
        gra = np.squeeze(gra)
        max_grad = np.maximum(gra, max_grad)
    return max_grad

def GradRandMinimizer(model, inp, lab, eps, iterations=1, transpose_grad=True):
    # Run attack for iterations to maximize the model gradient
    min_grad = np.squeeze(copy.deepcopy(inp.detach().numpy()) + 100000000.0)
    orig_inp = copy.deepcopy(inp.detach().numpy())
    eps = np.ones(orig_inp.shape)*eps
    zeros = np.zeros(orig_inp.shape)
    for i in range(iterations):
        adv = orig_inp + np.random.uniform(zeros, eps)
        gra = InputGrad(model, torch.Tensor(adv), lab)
        gra = gra.detach().numpy()
        if(transpose_grad):
            gra = gra.T
        gra = np.squeeze(gra)
        min_grad = np.minimum(gra, min_grad)
    return min_grad

def GradRandBounds(model, inp, lab, eps, iterations=10, transpose_grad=True):
    ma = GradRandMaximizer(model, inp, lab, eps, iterations, transpose_grad)
    mi = GradRandMinimizer(model, inp, lab, eps, iterations, transpose_grad)
    return torch.Tensor(mi), torch.Tensor(ma)

def GradRandRegularizer(model, inp, lab, eps, gam, nclasses=10, iterations=32):
    # Run attack to maximize the model gradient
    randmax_grad = GradRandMaximizer(model, data, target, eps, iterations=iterations)
    # Run attack to minimize the model gradient
    randmin_grad = GradRandMinimizer(model, data, target, eps, iterations=iterations)
    # Return regularizer value
    return torch.mean(randmax_grad - randmin_grad) * len(inp)


"""
Grad Attack Methods - Testing with Attacks rather than intervals
"""
def GradAttackMaximizer(model, data, target, eps=0.1):
    """
    Can only be run one input at a time for now
    """
    index = int(target[0])
    targets = torch.squeeze(target.repeat(1, model.in_dim))
    y = torch.squeeze(model(data))[index]
    data = torch.squeeze(data)
    mods = eps * torch.eye(data.shape[0])
    ey_p = torch.Tensor((data + mods).detach())
    ey_0 = torch.Tensor((data + (mods * 0.0)).detach())
    ey_n = torch.Tensor((data - mods).detach())
    targets = torch.squeeze(targets)
    val_p = torch.diagonal(_INGRAD(model, ey_p, targets))
    val_0 = torch.diagonal(_INGRAD(model, ey_0, targets))
    val_n = torch.diagonal(_INGRAD(model, ey_n, targets))
    add_indexes = (val_p > val_n).detach().numpy().astype(int)
    sub_indexes = (val_n > val_p).detach().numpy().astype(int)
    return torch.Tensor(eps*(add_indexes - sub_indexes))

def GradAttackMinimizer(model, data, target, eps=0.1):
    """
    Can only be run one input at a time for now
    """
    index = int(target[0])
    targets = torch.squeeze(target.repeat(1, model.in_dim))
    y = torch.squeeze(model(data))[index]
    data = torch.squeeze(data)
    mods = eps * torch.eye(data.shape[0])
    ey_p = torch.Tensor((data + mods).detach())
    ey_0 = torch.Tensor((data + (mods * 0.0)).detach())
    ey_n = torch.Tensor((data - mods).detach())
    targets = torch.squeeze(targets)
    val_p = torch.diagonal(_INGRAD(model, ey_p, targets))
    val_0 = torch.diagonal(_INGRAD(model, ey_0, targets))
    val_n = torch.diagonal(_INGRAD(model, ey_n, targets))
    add_indexes = (val_p < val_n).detach().numpy().astype(int)
    sub_indexes = (val_n < val_p).detach().numpy().astype(int)
    return torch.Tensor(eps*(add_indexes - sub_indexes))

def GradAttackBounds(model, data, target, eps=0.01,  nclasses=10):
    g = torch.squeeze(InputGrad(model, data, target, nclasses))
    H_max = GradAttackMaximizer(model, data, target, eps=eps)
    H_min = GradAttackMinimizer(model, data, target, eps=eps)
    adv_max = (data + H_max).detach().numpy()
    adv_min = (data + H_min).detach().numpy()   
    g_max = InputGrad(model, torch.Tensor(adv_max), target, nclasses)
    g_min = InputGrad(model, torch.Tensor(adv_min), target, nclasses)
    g_max = torch.squeeze(g_max)
    g_min = torch.squeeze(g_min)
    return torch.minimum(g_min, g), torch.maximum(g_max, g)

def GradAttackRegularizer(model, inp, lab, eps, gam, nclasses=10, iterations=32):
    # Run attack to maximize the model gradient
    randmax_grad = GradAttackMaximizer(model, data, target, eps, iterations=iterations)
    # Run attack to minimize the model gradient
    randmin_grad = GradAttackMinimizer(model, data, target, eps, iterations=iterations)
    # Return regularizer value
    return torch.mean(randmax_grad - randmin_grad) * len(inp)




def _INGRAD2(model, data, target, nclasses=10):
    device = torch.device("cpu")
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    try:
        output = torch.nn.Softmax(dim=1)(output)
    except:
        output = torch.nn.Softmax(dim=0)(output)
        output = output[None, :]
    #target = torch.nn.functional.one_hot(target, num_classes=nclasses)
    target = torch.unsqueeze(target, 0)
    print(output, target)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    print(data_grad)
    return data_grad

def SmoothGrad():
    print("Unimplimented")
    return None

# I have taken it from here: https://git.tu-berlin.de/gmontavon/lrp-tutorial
def FCN_LRP(model, data, target, nclasses=10):
    W = []
    B = []
    p = [t for t in model.parameters()]
    for i in range(len(p)):
        if(i%2==0):
            W.append(p[i].T)
        else:
            B.append(p[i])
    T = torch.nn.functional.one_hot(target, num_classes=nclasses)
    #W = [model.l1.weight.T, model.l2.weight.T]
    #B = [model.l1.bias, model.l2.bias]
    L = len(W)
    A = [data]+[None]*L
    for l in range(L):
        val = np.dot(A[l].detach().numpy(), W[l].detach().numpy())+B[l].detach().numpy()
        A[l+1] = torch.Tensor(np.maximum(0, val))

    R = [None]*L + [A[L]*T]#(T[:,None]==numpy.arange(10))]
    #print(A[L])
    #print("R: ",R)
    def rho(w,l):  return w + [None,0.1,0.0,0.0][l] * torch.Tensor(np.maximum(0,w.detach().numpy()))
    def incr(z,l): return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9
    for l in range(1,L)[::-1]:

        w = rho(W[l],l)
        b = rho(B[l],l)

        z = incr(torch.matmul(A[l], w)+b,l)    # step 1
        s = R[l+1] / z               # step 2
        c = torch.matmul(s, w.T)               # step 3
        R[l] = A[l]*c                # step 4
    w  = W[0]
    wp = torch.Tensor(np.maximum(0,w.detach().numpy()))
    wm = torch.Tensor(np.minimum(0,w.detach().numpy()))
    lb = (A[0]*0)-1
    hb = (A[0]*0)+1
    z = torch.matmul(A[0], w)-torch.matmul(lb,wp)-torch.matmul(hb,wm)+1e-9        # step 1
    s = R[1]/z                                        # step 2
    c,cp,cm  = torch.matmul(s,w.T),torch.matmul(s,wp.T),torch.matmul(s,wm.T)     # step 3
    R[0] = A[0]*c-lb*cp-hb*cm                         # step 4
    return R[0].detach().numpy()



def ReliabilityMap(model, inp, lab, eps, gam, nclasses=10):
    weights = [t.detach().numpy() for t in model.parameters()]
    logit_l, logit_u, inter_l, inter_u = IntervalBoundForward(weights, inp, eps, gam)
    v1 = torch.nn.functional.one_hot(lab, num_classes=nclasses)
    v2 = 1 - torch.nn.functional.one_hot(lab, num_classes=nclasses) 
    worst_case = torch.add(torch.multiply(v2, logit_u), torch.multiply(v1, logit_l))
    best_case = torch.add(torch.multiply(v1, logit_u), torch.multiply(v2, logit_l))
    y_l = torch.nn.Softmax(dim=1)(worst_case)
    y_u = torch.nn.Softmax(dim=1)(best_case)
    min_grad, max_grad = IntervalBoundBackward(model, y_l, y_u, v1, weights, inter_l, inter_u, gam)
    return 1/(max_grad-min_grad)

def ComputeReliability_Inp(model, inp, lab, max_eps, disc=20, transpose_grad=True):
    max_mask = np.squeeze(copy.deepcopy(inp.detach().numpy()) * 0.0)
    for eps in np.linspace(0.0, max_eps, disc):
        gra = InputGrad(model, inp, lab)
        gra =  gra.detach().numpy()
        min_grad, max_grad = GradCertBounds(model, inp, lab, eps, 0.00)
        min_grad, max_grad = min_grad.detach().numpy(), max_grad.detach().numpy()
        if(transpose_grad):
            min_grad = min_grad.T; max_grad = max_grad.T; gra = gra.T
        mask = (np.abs(np.asarray(gra).reshape(28,28)) >= np.asarray(max_grad-min_grad)).astype('float')
        mask *= eps
        mask = np.squeeze(mask)
        max_mask = np.maximum(mask, max_mask)
    return max_mask

def ComputeReliability_Mod(model, inp, lab, max_gam, disc=20, transpose_grad=True):
    max_mask = np.squeeze(copy.deepcopy(inp.detach().numpy()) * 0.0)
    for gam in np.linspace(0.0, max_gam, disc):
        gra = InputGrad(model, inp, lab)
        gra =  gra.detach().numpy()
        min_grad, max_grad = GradCertBounds(model, inp, lab, 0.00, gam)
        min_grad, max_grad = min_grad.detach().numpy(), max_grad.detach().numpy()
        if(transpose_grad):
            min_grad = min_grad.T; max_grad = max_grad.T; gra = gra.T
        mask = (np.abs(np.asarray(gra).reshape(28,28)) >= np.asarray(max_grad-min_grad)).astype('float')
        mask *= gam
        mask = np.squeeze(mask)
        max_mask = np.maximum(mask, max_mask)
    return max_mask


def MeaningfulProp():
    print("Unimplimented")
    return None
