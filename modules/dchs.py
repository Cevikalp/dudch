import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import geotorch

class dchs_loss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=285, feat_dim=2, precalc_centers=None,margin=1.0):
        super(dchs_loss, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        if(precalc_centers):
            precalculated_centers = np.load(precalc_centers)
        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,requires_grad=True))
            if(precalc_centers):
                self.centers.copy_(torch.from_numpy(precalculated_centers[:,0,:]))
                print('Centers loaded.')

    def forward(self, labels, x, x_out, ramp=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)
        intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()
        intraclass_loss = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)

        centers_dist_inter = (intraclass_distances.repeat(self.num_centers,1).t() - inlier_distmat)

        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes))

        interclass_loss_triplet = (1/(self.num_centers*batch_size*2.0))*((self.margin+centers_dist_inter).clamp(min=0)*mask).sum()
        if x_out!=None:
            batch_size_out = x_out.size(0)
            outlier_distmat = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(batch_size_out, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size_out).t()
            outlier_distmat.addmm_(x_out, self.centers.t(),beta=1,alpha=-2)
            outlier_corresponding_multi_distances = outlier_distmat.index_select(1,labels.long())
      
            if ramp:
                hinge_part = (self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0.).clamp(max=60.)
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*hinge_part.sum()
            else:
                ## WORKING PART DO NOT REMOVE..
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*((self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0)).sum()

            return intraclass_loss, interclass_loss_triplet, outlier_triplet_multi_loss
            # return intraclass_loss, interclass_loss_triplet, outlier_triplet_loss
        else:
            return intraclass_loss, interclass_loss_triplet, None
        
class NirvanaOpenset_loss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=285, feat_dim=2, precalc_centers=None, margin=1.0, Expand=50):
        super(NirvanaOpenset_loss, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.E = num_classes
        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]

        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, requires_grad=False))
            if(precalc_centers):
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')

        
    def forward(self, labels, x, x_out, ramp=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)
        intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()
        intraclass_loss = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)
    
        centers_dist_inter = (intraclass_distances.repeat(self.num_centers,1).t() - inlier_distmat)
    
        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes))
    
        interclass_loss_triplet = (1/(self.num_centers*batch_size*2.0))*((self.margin+centers_dist_inter).clamp(min=0)*mask).sum()
        if x_out!=None:
            batch_size_out = x_out.size(0)
            outlier_distmat = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(batch_size_out, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size_out).t()
            outlier_distmat.addmm_(x_out, self.centers.t(),beta=1,alpha=-2)
            outlier_corresponding_multi_distances = outlier_distmat.index_select(1,labels.long())
      
            if ramp:
                hinge_part = (self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0.).clamp(max=60.)
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*hinge_part.sum()
            else:
                ## WORKING PART DO NOT REMOVE..
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*((self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0)).sum()
    
            return intraclass_loss, interclass_loss_triplet, outlier_triplet_multi_loss
            # return intraclass_loss, interclass_loss_triplet, outlier_triplet_loss
        else:
            return intraclass_loss, interclass_loss_triplet, None
        
def FindCenters(k, E=1):
    """
    Calculates "k+1" equidistant points in R^{k}.
    Args:
        k (int) dimension of the space
        E (float) expand factor 
    Returns: 
        Centers (np.array) equidistant positions in R^{k}, shape (k+1 x k)
    """
    
    Centers = np.empty((k+1, k), dtype=np.float32)
    CC = np.empty((k,k), dtype=np.float32)
    Unit_Vector = np.identity(k)
    c = -((1+np.sqrt(k+1))/np.power(k, 3/2))
    CC.fill(c)
    d = np.sqrt((k+1)/k)
    DU = d*Unit_Vector 
    Centers[0,:].fill(1/np.sqrt(k))
    Centers[1:,:] = CC + DU
    
    # Calculate and Check Distances
    # Distances = np.empty((k+1,k), dtype=np.float32)
    # for k, rows in enumerate(Centers):
    #     Distances[k,:] = np.linalg.norm(rows - np.delete(Centers, k, axis=0), axis=1)
    # # print("Distances:",Distances)    
    # assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal" 
    return Centers*E

class nirvana_hypersphere(nn.Module):
    """Center loss with subcenters added.
   
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=3, feat_dim=2, normalization = True, precalc_centers=None, margin=12.0, Expand=48, kernel_width=0.0001):
        super(nirvana_hypersphere, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.mse =  nn.MSELoss()
        self.normalization = normalization
        self.E = Expand
        self.t = kernel_width
        #self.interclass_margin = interclass_margin
        
        with torch.no_grad():
            # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim,device=self.device,requires_grad=False))
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim), requires_grad=True)
            precalculated_centers = torch.mul(torch.nn.functional.normalize(self.centers),self.E)
            if(precalc_centers):
                # precalculated_centers*=10
                self.centers.copy_(precalculated_centers)
                print('Centers loaded.')
        #self.margin = torch.mul(torch.mul(self.num_classes-1, self.E), 2).item()
        self.margin = margin
    def forward(self, labels, x, x_out, ramp=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            nearest (bool): assign subcenter according to labels or nearest
        """
        batch_size = x.size(0)
        # x = torch.mul(nn.functional.normalize(x), self.E)
        
        with torch.no_grad():
            if self.normalization:
                self.centers.copy_(torch.mul(torch.nn.functional.normalize(self.centers),self.E))
        centers_batch = self.centers.view(-1,self.feat_dim).index_select(0, labels.long())
        
        #centers_norms = torch.norm(self.centers,dim=1)
        #print(centers_norms)
               
        centers_distance = 1/(torch.pow(torch.cdist(self.centers, self.centers),2)+1) 
        centers_distance.fill_diagonal_(0)
        uniform_loss = torch.sum(centers_distance)

        uniform_loss_term = torch.exp(-self.t*torch.pow(torch.cdist(self.centers, self.centers),2))
        uniform_loss2 = torch.log(torch.sum(uniform_loss_term))
       
       # uniform_loss = torch.div(1,dist_centers)
        
        intraclass_loss = self.mse(x, centers_batch)
        # total_loss = torch.add(intraclass_loss, intraclass_loss, alpha=0.5)  
        xnormalized = torch.mul(nn.functional.normalize(x), self.E)
      

         
        inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)
        intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()
        intraclass_loss = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)
        
        centers_dist_inter = (intraclass_distances.repeat(self.num_centers,1).t() - inlier_distmat)
        # outlierdaki her bir ornegi her sınıf için de kullan
        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes))
        #batch_size'a bölmeyi deneyebilirsin. (1/mask.count_nonzero())*
        interclass_loss_triplet = (1/(self.num_centers*batch_size*2.0))*((self.margin+centers_dist_inter).clamp(min=0)*mask).sum()
        if x_out!=None:
            batch_size_out = x_out.size(0)
            outlier_distmat = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(batch_size_out, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size_out).t()
            outlier_distmat.addmm_(x_out, self.centers.t(),beta=1,alpha=-2)
            outlier_corresponding_multi_distances = outlier_distmat.index_select(1,labels.long())
      
            if ramp:
                hinge_part = (self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0.).clamp(max=60.)
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*hinge_part.sum()
            else:
                ## WORKING PART DO NOT REMOVE..
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*((self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0)).sum()
    
            return intraclass_loss, interclass_loss_triplet, uniform_loss2, outlier_triplet_multi_loss
            # return intraclass_loss, interclass_loss_triplet, outlier_triplet_loss
        else:
            return intraclass_loss, interclass_loss_triplet, uniform_loss, None
        

def get_l2_pred(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits = 1/(1+distmat)
            return pred, logits
        else:
            return pred

def get_l2_pred_b9(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits_b9 = 1/(1+F.normalize(distmat,p=2))
            logits = 1/(1+distmat)
            return pred, logits, logits_b9
        else:
            return pred

def accuracy_l2(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)  