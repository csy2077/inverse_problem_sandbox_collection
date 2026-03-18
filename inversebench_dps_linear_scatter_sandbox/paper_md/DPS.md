

# DIFFUSION POSTERIOR SAMPLING FOR GENERAL  NOISY INVERSE PROBLEMS 

Hyungjin Chung1,2,Jeongsol Kim1,Michal T.Mccann2,Marc L.Klasky& Jong Chul Ye11KAIST,2 Los Alamos National Laboratory 



{hj.chung, jeongsol, jong.ye}@kaist.ac.kr, {mccann, mklasky}@lanl.gov 

## ABSTRACT 

Diffusion models have been recently studied as powerful generative inverse problem solvers, owing to their high quality reconstructions and the ease of combining existing iterative solvers. However, most works focus on solving simple linear inverse problems in noiseless settings, which significantly under-represents the complexity of real-world problems. In this work, we extend diffusion solvers to efficiently handle general noisy (non)linear inverse problems via approximation of the posterior sampling. Interestingly, the resulting posterior sampling scheme is a blended version of diffusion sampling with the manifold constrained gradient without a strict measurement consistency projection step, yielding a more desirable generative path in noisy settings compared to the previous studies. Our method demonstrates that diffusion models can incorporate various measurement noise statistics such as Gaussian and Poisson, and also efficiently handle noisy nonlinear inverse problems such as Fourier phase retrieval and non-uniform deblurring. Code is available at https://github.com/DpS2022/diffusion-posterior-sampling.

## 1 INTRODUCTION 

Diffusion models learn the implicit prior of the underlying data distribution by matching the gradient of the log density (i.e. Stein score;$\nabla_{\pmb{x}}\log p(\pmb{x}))$ ) (Song et al., 2021b). The prior can be leveraged when solviivreproblm,whch imtocov om te auem,d throuh t forward measurement operator A and the detector noise n. When we know such forward models,one can incorporate the gradient of the log likelihood (i.e.$\nabla_{\pmb{x}}\log p(\pmb{y}|\pmb{x}))$ in order to sample from the posterior distribution $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}}})$ 1. While this looks straightforward, the likelihood term is in fact analytically intractable in terms of diffusion models, due to their dependence on time t. Due to its intractability, one often resorts to projections onto the measurement subspace (Song et al., 2021b;Chung et al., 2022b; Chung & Ye, 2022; Choi et al., 2021). However, the projection-type approach fails dramatically when 1) there is noise in the measurement, since the noise is typically amplified during the generative process due to the ill-posedness of the inverse problems; and 2) the measurement process is nonlinear.



One line of works that aim to solve noisy inverse problems run the diffusion in the spectral domain (Kawar et al., 2021; 2022) so that they can tie the noise in the measurement domain into the spectral domain via singular value decomposition (SVD). Nonetheless, the computation of SVD is costly and even prohibitive when the forward model gets more complex. For example, Kawar et al.(2022) only considered seperable Gaussian kernels for deblurring, since they were restricted to the family of inverse problems where they could effectively perform the SVD. Hence, the applicability of such methods is restricted, and it would be useful to devise a method to solve noisy inverse problems without the computation of SVD. Furthermore, while diffusion models were applied to various inverse problems including inpainting (Kadkhodaie & Simoncelli, 2021; Song et al., 2021b; Chung et al.,2022b; Kawar et al., 2022; Chung et al., 2022a), super-resolution (Kadkhodaie & Simoncelli, 2021;Choi et al., 2021; Chung et al., 2022b; Kawar et al., 2022), colorization (Song et al., 2021b; Kawar et al., 2022; Chung et al., 2022a), compressed-sensing MRI (CS-MRI) (Song et al., 2022; Chung &Ye, 2022; Chung et al., 2022b), computed tomography (CT) (Song et al., 2022; Chung et al., 2022a),etc., to our best knowledge, all works so far considered linear inverse problems only, and have not explored nonlinear inverse problems.



<div style="text-align: center;"><img src="imgs/img_in_image_box_121_106_1072_577.jpg" alt="Image" width="77%" /></div>


<div style="text-align: center;">Figure 1: Solving noisy linear, and nonlinear inverse problems with diffusion models. Our reconstruction results (right) from the measurements (left) are shown. </div>


In this work, we devise a method to circumvent the intractability of posterior sampling by diffusion models via a novel approximation, which can be generally applied to noisy inverse problems.Specifically, we show that our method can efficiently handle both the Gaussian and the Poisson measurement noise. Also, our framework easily extends to any nonlinear inverse problems, when the gradients can be obtained through automatic differentiation. We further reveal that a recently proposed method of manifold constrained gradients (MCG) (Chung et al., 2022a) is a special case of the proposed method when the measurement is noiseless. With a geometric interpretation, we further show that the proposed method is more likely to yield desirable sample paths in noisy setting than the previous approach (Chung et al., 2022a). In addition, the proposed method fully runs on the image domain rather than the spectral domain, thereby avoiding the computation of SVD for effi cient implementation. With extensive experiments including various inverse problems—inpainting,super-resolution, (Gaussian/motion/non-uniform) deblurring, Fourier phase retrieval—we show that our method serves as a general framework for solving general noisy inverse problems with superior quality (Representative results shown in Fig. 1).



## 2 BACKGROUND 

### 2.11 SCORE-BASED DIFFUSIONMODELS 

Diffusion models define the generative process as the reverse of the noising process. Specifically,Song et al. (2021b) defines the Itô stochastic differential equation (SDE) for the data noising process (i.e. forward SDE)$\begin{aligned}{\mathbf{\mathit{x}}(t),\:t\in[0,T],\:\mathbf{\mathit{x}}(t)\in\mathbb{R}^{d}\:\forall t}\\ \end{aligned}$ inthefollowingform1

$$d\mathbf{\mathit{x}}=-\frac{\beta(t)}{2}\mathbf{\mathit{x}}d t+\sqrt{\beta(t)}d\mathbf{\mathit{w}},$$

where $\beta(t):\mathbb{R}\to\mathbb{R}>0$ is the noise schedule of the process, typically taken to be monotonically increasing linear function of t (Ho et al., 2020), and w is the standard d−dimensional Wiener process.The data distribution is defined when $t=0,\mathrm{i.e.}\;\boldsymbol{x}(0)\sim p_{\mathrm{d a t a}}$ , and a simple, tractable distribution (e.g.isotropic Gaussian) is achieved when $t=T,\mathrm{i.e.}\widehat{\boldsymbol{x}(T)}\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})$ 



Our aim is to recover the data generating distribution starting from the tractable distribution, which can be achieved by writing down the corresponding reverse SDE of (1) (Anderson, 1982):

$$d\mathbf{\mathit{x}}=\left[-\frac{\beta(t)}{2}\mathbf{\mathit{x}}-\beta(t)\nabla_{\mathbf{\mathit{x}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathit{x}}_{t})\right]d t+\sqrt{\beta(t)}d\bar{\mathbf{\mathit{w}}},$$

where dt corresponds to time running backward and dw to the standard Wiener process running backward. The drift function now depends on the time-dependent score function $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}_{t})$ which is approximated by a neural network $\pmb{s}_{\theta}$ trained with denoising score matching (Vincent, 2011):

$$\theta^{*}=\mathop{\operatorname{a r g}\operatorname*{m i n}}_{\theta}\mathbb{E}_{t\sim U(\varepsilon,1),\mathbf{x}(t)\sim p(\mathbf{x}(t)|\mathbf{x}(0)),\mathbf{x}(0)\sim p_{\mathsf{l i s s}}}\left[\|\mathbf{s}_{\theta}(\mathbf{x}(t),t)-\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p(\mathbf{x}(t)|\mathbf{x}(0))\|_{2}^{2}\right] ,(3)$$

where $\varepsilon\simeq0$ is a small positive constant. Once $\theta^{*}$ is acquired through (3), one can use the approximation $\nabla_{\mathbf{x}_{t}}$ log $p_{t}(\hat{\mathbf{\mathit{x}}_{t}})\simeq\mathbf{\mathit{s}}_{\theta^{*}}(\mathbf{\mathit{x}}_{t},t)$ as a plug-in es $\dot{\mathrm{state}}^2$ to replace the score function in (2).Discretization of (2) and solving using, e.g. Euler-Maruyama discretization, amounts to sampling from the data distribution $p({\pmb x})$ |, the goal of generative modeling.



Throughout the paper, we adopt the standard VP-SDE (i.e. ADM of Dhariwal & Nichol (2021) or Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020)), where the reverse diffusion variance which we denote by $\tilde{\sigma}(t)$ is learned as in Dhariwal & Nichol (2021). In discrete settings (e.g. in the algorithm) with N bins, we define $\begin{aligned}{\mathbf{x}_{i}\triangleq\mathbf{x}(t T/N),\beta_{i}\triangleq\beta(t T/N)}\\ \end{aligned}$ , and subsequently $\begin{array}{l}{\alpha_{i}\triangleq1-\beta_{i},\bar{\alpha}_{i}\triangleq\prod_{j=1}^{i}\alpha_{i}}\\ \end{array}$ following Ho et al. (2020).



### 2.2 INVERSE PROBLEM SOLVING WITH DIFFUSION MODELS 

For various scientific problems, we have a partial measurement y that is derived from x. When the mapping $\mathbf{\mathit{x}}\mapsto\mathbf{\mathit{y}}$ is many-to-one, we arrive at an ill-posed inverse problem, where we cannot exactly retrieve x. In the Bayesian framework, one utilizes $p({\pmb x})$ as the prior, and samples from the posterior $p(\pmb{x}|\pmb{y})$ , where the relationship is formally established with the Bayes'rule:$p(\mathbf{\mathbf{\mathbf{\mathit{x}}}}|\mathbf{\mathbf{\mathbf{\mathit{y}}}})=$ $p(\mathbf{\mathbf{\mathit{y}}}|\mathbf{\mathbf{\mathit{x}}})p(\mathbf{\mathbf{\widehat{\mathit{x}}}})/p(\mathbf{\mathbf{\widehat{\mathit{y}}}})$ . Leveraging the diffusion model as the prior it is straightforward to modif $(2)$ arrive at the reverse diffusion sampler for sampling from the posterior distribution:

$$d\mathbf{\mathit{x}}=\left[-\frac{\beta(t)}{2}\mathbf{\mathit{x}}-\beta(t)(\nabla_{\mathbf{\mathit{x}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathit{x}}_{t})+\nabla_{\mathbf{\mathit{x}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathit{y}}|\mathbf{\mathit{x}}_{t}))\right]d t+\sqrt{\beta(t)}d\bar{\mathbf{\mathit{w}}},$$

where we have used the fact that 

$$\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{x}_{t}|\mathbf{y})=\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{x}_{t})+\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{y}|\mathbf{x}_{t}).$$

In (4), we have two terms that should be computed: the score function $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{t})$ , and the likelihood $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}_{t})$ 1. To compute the former term involving $p_{t}(\pmb{x})$ ), we can simply use the pre-trained score function $\pmb{s}_{\theta^{*}}$ . However, the latter term is hard to acquire in closed-form due to the dependence on the time $t,$ as there only exists explicit dependence between y and $\boldsymbol{x}_{0}$ .

Formally, the general form of the forward model3 can be stated as 

$$\mathbf{\mathit{y}}=\mathcal{A}(\mathbf{\mathit{x}}_{0})+\mathbf{\mathit{n}},\quad\mathbf{\mathit{y}},\mathbf{\mathit{n}}\in\mathbb{R}^{n},\:\mathbf{\mathit{x}}_{0}\in\mathbb{R}^{d},$$

where $\mathcal{A}(\cdot){:}\mathbb{R}^{d}\mapsto\mathbb{R}^{n}$ is the forward measurement operator and n is the measurement noise. In the case of white Gaussian noise,$\mathbf{\mathit{n}}\sim\mathcal{N}(0,\sigma^{2}\mathbf{\mathit{I}})$ 1. In explicit form,$p(\mathbf{\mathbf{\mathit{y}}}|\mathbf{\mathbf{\mathit{x}}}_{0})\sim\mathcal{N}(\mathbf{\mathbf{\mathit{y}}}|\mathcal{A}(\mathbf{\mathbf{\mathit{x}}}_{0}),\sigma^{2}\mathbf{\mathbf{\mathit{I}}})$ However, there does not exist explicit dependency between y and $\boldsymbol{x}_{t}$ , as can be seen in the probabilistic graph from Fig. 2, where the blue dotted line remains unknown.



In order to circumvent using the likelihood term directly, alternating projections onto the measurement subspace is a widely used strategy (Song et al., 202ib; Chung & Ye, 2022; Chung et al., 2022b).Namely, one can disregard the liklihood term in (4), and frst take an unconditional update with (2),and then take a projection step such that measurement consistency can be met, when assuming $\mathbf{n}\simeq0$ (Kadkhodaie & Simoncelli, 2021) proposes to use a coarse-to-fine gradien update from the likelihood obtained om teTwedie nid timate.Anothlieof work Jalaltal,2021)lves liear inverse problems where $\mathcal{A}(\mathbf{\mathit{x}})\triangleq\mathbf{\mathit{A}}\mathbf{\mathit{x}}$ : and utilizes an approximation $\begin{array}{r l}{\nabla_{\pmb{x}_{t}}\log p_{t}(\pmb{y}|\pmb{x})\simeq\frac{\pmb{A}^{H}(\pmb{y}-\pmb{A}\pmb{x})}{\sigma^{2}}}\\ \end{array}$ which is obtained when n is assumed to be Gaussian noise with variance $\sigma^{2}$ . Nonetheless, the equation is only correct when $t=0$ , while being wrong at all other noise levels that are actually used in the generative process.The incorrectness is counteracted by a heuristic of assuming higher evls 

of noise as $t\to T$ , such that $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}$ log $p_{t}(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}})\simeq\textstyle\frac{\mathbf{\mathbf{\mathbf{\mathit{A}}}}^{H}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}-\mathbf{\mathbf{\mathbf{\mathit{A}}}}\mathbf{\mathbf{\mathbf{\mathit{x}}}})}{\sigma^{2}+\gamma_{t}^{2}}$ ,where $\{\gamma_{t}\}_{t=1}^{T}$ are hyperparameters.While both lines of works aim to perform posterior sampling given the measurements and empirically work well on noiseless inverse problems, it should be noted that 1) they do not provide means to handle measurement noise, and 2) using such methods to solve nonlinear inverse problems either fails to work or is not straightforward to implement. The aim of this paper is to take a step toward a more general inverse problem solver, which can address noisy measurements and also scales effectively to nonlinear inverse problems.



## 3 DIFFUSioN POSTeRIOR SAMPLING (DPS)

### 3.1 APPROXIMATION OF THE LIKELIHOOD 

Recall that no analytical formulation for $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{t})$ exists.In order to exploit the measurement model $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{0})$ 1, we factorize $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{t})$ as follows:

$$\begin{aligned}{p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})}&{{}=\int p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0},\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})p(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})d\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}}\\ {}&{{}=\int p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0})p(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})d\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0},}\\ \end{aligned}$$

where the second equality comes from that y and $\boldsymbol{x}_{t}$ are conditionally independent on ${\pmb x}_{0}$ , as shown in Fig. 2. Here,$p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}_{t})$ , as was shown with blue dotted lines in Fig. 2,is intractable in general. Note however, that for the case 

$$x_{t}$$

<div style="text-align: center;">Figure 2: Probabilistic graph. Black solid line: tractable, blue dotted line: intractable in general. </div>


of diffusion models such as VP-SDE or DDPM, the forward diffusion can be simply represented by 

$$\mathbf{\mathit{x}}_{t}=\sqrt{\bar{\alpha}(t)}\mathbf{\mathit{x}}_{0}+\sqrt{1-\bar{\alpha}(t)}\mathbf{\mathit{z}},\qquad\mathbf{\mathit{z}}\sim\mathcal{N}(\mathbf{0},\mathbf{\mathit{I}}),$$

so that we can obtain the specialized representation of the posterior mean as shown in Proposition 1through the Tweedie's approach (Efron, 2011; Kim & Ye, 2021). Detailed derivations can be found in Appendix A.



Proposition 1. For the case of VP-SDE or DDPM sampling,$p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{t})$ has the unique posterior mean at 



$$\hat{\pmb x}_{0}:=\mathbb{E}[{\pmb x}_{0}|{\pmb x}_{t}]=\frac{1}{\sqrt{\bar{\alpha}(t)}}({\pmb x}_{t}+(1-\bar{\alpha}(t))\nabla_{{\pmb x}_{t}}\operatorname{l o g}p_{t}({\pmb x}_{t}))$$

Remark 1..$B y$ replacing $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}_{t}}\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}_{t})$ in (9) with the score estimate $\mathbf{s}_{\theta^{*}}(\mathbf{x}_{t})$ , we can approximate the posterior mean from $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}_{t})$  as:

$$\hat{\mathbf{{x}}}_{0}\simeq\frac{1}{\sqrt{\bar{\alpha}(t)}}(\mathbf{{x}}_{t}+(1-\bar{\alpha}(t))\mathbf{{s}}_{\theta^{*}}(\mathbf{{x}}_{t},t)).$$

In fact, the result is closely related to the well established field of denoising. Concretely, consider the problem of retrieving the estimate of clean ${\pmb x}_{0}$ from the given Gaussian noisy $\boldsymbol{x}_{t}$ . A classic result of Tweedie's ormula (Robbins,192; ei,1981; Eo,201; im &$\gamma_{e}$ , 2021) states that one can retrieve the empiricalayesoptimal posterior mean $\hat{\boldsymbol{x}}_{0}$ using the formula in (10).

Given the posterior mean $\hat{\boldsymbol{x}}_{0}$ a to provide a tractable approximation for $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{t})$ such that one can use the surrogate function to maximize the likelihood—yielding approximate posterior sampling. Specifically, given the interpretation $p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{x}}}_{t})=\mathbb{E}_{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0}\sim p(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{t})}\left[p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{x}}}_{0})\right]$ from (7), we use the following approximation:

$$p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})\simeq\:p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\hat{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}),\quad\operatorname{w h e r e}\quad\hat{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}:=\mathbb{E}[\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{t}]=\mathbb{E}_{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}\sim p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})}\left[\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0}\right]$$

implying that the outer expectation of $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{0})$ over the posterior distribution is replaced with inner expectation of $x_{0}$ I so we need the following definition to quantify the approximation error:

Definition 1 (Jensen gap (Gao et al., 2017; Simic, 2008)). Let x be a random variable with distribution $p({\pmb x})$ . For some function that may or may not be convex,the Jensen gap is defined as 



$$\mathcal{J}(f,\mathbf{\mathit{x}}\sim p(\mathbf{\mathit{x}}))=\mathbb{E}[f(\mathbf{\mathit{x}})]-f(\mathbb{E}[\mathbf{\mathit{x}}]),$$

where the expectation is taken over $p(\pmb x)$ ).

Algorithm 1 DPS - Gaussian Algorithm 2 DPS - Poisson 
Require:$N,\boldsymbol{y},\{\zeta_{i}\}_{i=1}^{N},\{\tilde{\sigma}_{i}\}_{i=1}^{N}$ Require:$N,\boldsymbol{y},\{\zeta_{i}\}_{i=1}^{N},\{\tilde{\sigma}_{i}\}_{i=1}^{N}$ 
1:$\mathbf{\mathit{x}}_{N}\sim\mathcal{N}(\mathbf{0},\mathbf{\mathit{I}})$ 1:$\bar{\mathbf{x}}_{N}\sim\mathcal{N}(\mathbf{0},\bar{\mathbf{}I})$ 

2: for $i=N-1\mathbf{t o}0\mathbf{d o}$ 2:$\mathbf{f o r}\:i=N-1\:\mathbf{t o}\:0\:\mathbf{d o}$ 
3:$\hat{\mathbf{s}}\gets\mathbf{s}_{\theta}(\mathbf{x}_{i},i)$ 3:$\hat{\mathbf{s}}\leftarrow\mathbf{s}_{\theta}(\mathbf{x}_{i},i)$ 

4:$\hat{\boldsymbol{x}}_{0}\leftarrow\frac{1}{\sqrt{\bar{\alpha}_{i}}}\left(\widehat{\boldsymbol{x}}_{i}+(1-\bar{\alpha}_{i})\hat{\boldsymbol{s}}\right)$ 4:$\hat{\mathbf{x}}_{0}\leftarrow\frac{1}{\sqrt{\bar{\alpha}_{i}}}(\widehat{\mathbf{x}_{i}}+(1-\bar{\alpha}_{i})\hat{\mathbf{s}})$ 
5:$\pmb{z}\sim\mathcal{N}(\mathbf{0},\pmb{I})$ 5:$\boldsymbol{z}\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})$ 

6:$\begin{array}{l}{\begin{array}{l}{\dot{\mathbf{x}_{i-1}^{\prime}}\leftarrow\frac{\sqrt{\bar{\alpha}_{i}}(1-\bar{\alpha}_{i-1})}{1-\bar{\alpha}_{i}}\mathbf{x}_{i}+\frac{\sqrt{\bar{\alpha}_{i-1}}\beta_{i}}{1-\bar{\alpha}_{i}}\hat{\mathbf{x}}_{0}+\tilde{\sigma}_{i}\mathbf{z}}\\ \end{array}}\\ \end{array}$ 6:$\begin{array}{l}{{\bf{x}}_{i-1}^{\prime}\leftarrow\frac{{\sqrt{{{\bar{\alpha}}_{i}}}\left({1-{{\bar{\alpha}}_{i-1}}}\right)}}{{1-{{\bar{\alpha}}_{i}}}}{{\bf{x}}_{i}}{+}\frac{{\sqrt{{{\bar{\alpha}}_{i-1}}}{\beta_{i}}}}{{1-{{\bar{\alpha}}_{i}}}}{{\hat{\bf{x}}}_{0}}{+}{\tilde{\sigma}_{i}}{\bf{z}}}\\ \end{array}$ 
7:$\mathbf{\mathit{x}}_{i-1}\leftarrow\mathbf{\mathit{x}}_{i-1}^{\prime}-\widetilde{\zeta}_{i}\nabla_{\mathbf{\mathit{x}}_{i}}\|\mathbf{\mathit{y}}-\mathcal{A}(\hat{\mathbf{\mathit{x}}}_{0})\|_{2}^{2}$ 7:$\mathbf{\mathit{x}}_{i-1}\leftarrow\mathbf{\mathit{x}}_{i-1}^{\prime}-\zeta_{i}\nabla_{\mathbf{\mathit{x}}_{i}}\|\mathbf{\mathit{y}}-\mathcal{A}(\hat{\mathbf{\mathit{x}}}_{0})\|_{\mathbf{\Lambda}}^{2}$ 
8: end for 8: end for 
9: return $\hat{\mathbf{x}}_{0}$ ,9: return x0

The following theorem derives the closed-form upper bound of the Jensen gap for the inverse problem from (6) when $\mathbf{\mathit{n}}\sim\mathcal{N}(0,\sigma^{2}\mathbf{\mathit{I}})$ 



Theorem 1. For the given measurement model (6) with $\mathbf{\mathit{n}}\sim\mathcal{N}(0,\sigma^{2}\mathbf{\mathit{I}})$ , we have 

$$p(\mathbf{\mathbf{\mathit{y}}}|\mathbf{\mathbf{\mathit{x}}}_{t})\simeq p(\mathbf{\mathbf{\mathit{y}}}|\hat{\mathbf{\mathbf{\mathit{x}}}}_{0}),$$

where the approximation error can be quantified with the Jensen gap, which is upper bounded by 

$$\mathcal{J}\leq\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|m_{1},$$

$$where \textstyle\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|:=\operatorname*{m a x}_{\pmb{x}}\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|\:{a n d}\:m_{1}:=\int\|\pmb{x}_{0}-\hat{\pmb{x}}_{0}\|p(\pmb{x}_{0}|\pmb{x}_{t})\:d\pmb{x}_{0}.$$

Remark 2. Note that $\|\nabla_{\boldsymbol{x}}\mathcal{A}(\boldsymbol{x})\|$ is finite in most of the inverse problems. This should not be confused with the ill-posedness of the inverse problems, which refers to the unboundedness of the inverse operator .$\mathcal{A}^{-1}$ . Accordingly,$\mathit{i f}m_{1}$ is also finite (which is the case for most of the distribution in practice), the Jensen gap in Theorem 1 can approach to 0 as $\sigma\;\to\;\infty$ ,suggesting that the approximation error reduces with higher measurement noise. This may explain why our DPS works well for noisy inverse problems. In addition, although we have specified the measurement distribution to be Gaussian, we can also determine the Jensen gap for other measurement distributions (e.g.Poisson) in an analogous fashion.



By leveraging the result of Theorem 1, we can use the approximate gradient of the log likelihood 

$$\nabla_{\mathbf{{x}}_{t}}\operatorname{l o g}p(\mathbf{{y}}|\mathbf{{x}}_{t})\simeq\nabla_{\mathbf{{x}}_{t}}\operatorname{l o g}p(\mathbf{{y}}|\hat{\mathbf{{x}}}_{0}),$$

where the latter is now analytically tractable, as the measurement distribution is given.

### 3.2 MODEL DEPENDENT LIKELIHOOD OF THE MEASUREMENT 

Note that we may have different measurement models $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{0})$ for each application. Two of the most common cases in inverse problems are the Gaussian noise and the Poisson noise. Here, we explore how our diffusion posterior sampling described above can be adapted to each case.

Gaussian noise. The likelihood function takes the form 

$$p(\mathbf{\mathit{\mathbf{y}}}|\mathbf{\mathit{\mathbf{x}}}_{0})=\frac{1}{\sqrt{(2\pi)^{n}\sigma^{2n}}}\operatorname{e x p}\left[-\frac{\|\mathbf{\mathit{\mathbf{y}}}-\mathcal{A}(\mathbf{\mathit{\mathbf{x}}}_{0})\|_{2}^{2}}{2\sigma^{2}}\right],$$

where n denotes the dimension of the measurement y. By differentiating $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{t})$ with respect to $\boldsymbol{x}_{t}$ using Theorem 1 and (15), we get 



$$\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{t}}\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}_{t})\simeq-\frac{1}{\sigma^{2}}\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}}\|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}}-\mathcal{A}(\hat{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}}_{0}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf}{\mathbf{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf}{\mathbf{\mathbf}{\mathbf{\mathbf}{\mathbf{x}}}{}}}}}}}}}}}}}}}}}}}}}}})\|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}}}}}}}_{2}$$

where we have explicitly denotd $\hat{\pmb x}_{0}{:=}\hat{\pmb x}_{0}({\pmb x}_{t})$ to emphasize that $\hat{\boldsymbol{x}}_{0}$ is a function of $\boldsymbol{x}_{t}$ . Consequently, taking the gradient $\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{t}}$ amounts to taking the backpropagation through the network.EM 

$$\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{x}_{t}|\mathbf{y})\simeq\mathbf{s}_{\theta^{*}}(\mathbf{x}_{t},t)-\rho\nabla_{\mathbf{x}_{t}}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_{0})\|_{2}^{2},$$

where.$\rho\triangleq1/\sigma^{2}$ is set as the step size.

\mathcal{M}_{4}

<div style="text-align: center;"><img src="imgs/img_in_chart_box_110_134_593_438.jpg" alt="Image" width="39%" /></div>


\mathcal{M}_{3}

\boldsymbol{y}=\mathcal{A}(\boldsymbol{x})

<div style="text-align: center;">(a) Geometry of Chung et al. (2022a)</div>


<div style="text-align: center;">Figure 3: Conceptual illustration of the geometries of two different diffusion processes. Our method prevents the sample from falling off the generative manifolds when the measurements are noisy.</div>


$$\mathcal{M}_{2}$$

$$\mathcal{M}_{1}$$

$$\mathcal{M}_{0}$$

<div style="text-align: center;">(b) Geometry of DPS </div>


given as Poisson noise. The likelihood function for the Poisson measurements under the i.i.d. assumption is 

$$p(\mathbf{\mathbf{\mathit{y}}}|\mathbf{\mathbf{\mathit{x}}}_{0})=\prod_{j=1}^{n}\frac{\left[\mathcal{A}(\mathbf{\mathbf{\mathit{x}}}_{0})\right]_{j}^{\mathbf{\mathbf{\mathit{y}}}_{j}}\operatorname{e x p}\left[[-\mathcal{A}(\mathbf{\mathbf{\mathit{x}}}_{0})]_{j}\right]}{\mathbf{\mathbf{\mathit{y}}}_{j}\:!},$$

h $j$ model can be approximated by a Gaussian distribution with very high accuracy4. Namely,

$$\begin{aligned}{p(\mathbf{\mathit{{y}}}|\mathbf{\mathit{{x}}}_{0})}&{{}\to\prod_{j=1}^{n}\frac{1}{\sqrt{2\pi[\mathcal{A}(\mathbf{\mathit{{x}}}_{0})]_{j}}}\operatorname{e x p}\left(-\frac{(\mathbf{\mathit{{y}}}_{j}-[\mathcal{A}(\mathbf{\mathit{{x}}}_{0})]_{j})^{2}}{2[\mathcal{A}(\mathbf{\mathit{{x}}}_{0})]_{j}}\right)}\\ {}&{{}\simeq\prod_{j=1}^{n}\frac{1}{\sqrt{2\pi\mathbf{\mathit{{y}}}_{j}}}\operatorname{e x p}\left(-\frac{(\mathbf{\mathit{{y}}}_{j}-[\mathcal{A}(\mathbf{\mathit{{x}}}_{0})]_{j})^{2}}{2\mathbf{\mathit{{y}}}_{j}}\right),}\\ \end{aligned}$$

where we have used the standard approximation for the shot noise model $[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0})]_{j}\simeq\mathbf{\mathbf{\mathbf{\mathit{y}}}}_{j}$ to arrive at the last equation (Kingston, 2013).Then, similar to the Gaussian case, by differentiation and the use of Theorem 1, we have that 



$$\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{t})\simeq-\rho\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\|\mathbf{\mathbf{\mathbf{\mathbf{y}}}}-\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{0})\|_{\mathbf{\mathbf{\mathbf{\mathbf{\Lambda}}}}}^{2},\quad[\mathbf{\mathbf{\mathbf{\Lambda}}}]_{i i}\triangleq1/2\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}}}_{i},$$

where $\|\boldsymbol{a}\|_{\boldsymbol{\Lambda}}^{2}\triangleq\boldsymbol{a}^{T}\boldsymbol{\Lambda}\boldsymbol{a}$ i, and we have included ρ to define the step size as in the Gaussian case. We can summarize our strategy for each noise model as follows:

$$\small\begin{aligned}{\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{x}_{t}|\mathbf{y})}&{{}\simeq s_{\theta^{*}}(\mathbf{x}_{t},t)-\rho\nabla_{\mathbf{x}_{t}}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_{0})\|_{2}^{2}}\\ {\nabla_{\mathbf{x}_{t}}\operatorname{l o g}p_{t}(\mathbf{x}_{t}|\mathbf{y})}&{{}\simeq s_{\theta^{*}}(\mathbf{x}_{t},t)-\rho\nabla_{\mathbf{x}_{t}}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_{0})\|_{\mathbf{A}}^{2}}\\ \end{aligned}$$

Incorporation of (16) or (21) into the usual ancestral sampling (Ho et al., 2020) steps leads to Algorithm 1,25. Here, we name our algorithm Diffusion Posterior Sampling (DPS), as we construct our method in order to perform sampling from the posterior distribution. Notice that unlike prior methods that limit their applications to linear inverse problems $\mathcal{A}(\mathbf{\mathit{x}})\triangleq\mathbf{\mathit{A}}\mathbf{\mathit{x}}$ , our method is fully general in that we can also use nonlinear operators $\hat{\mathcal{A}(\cdot)}$ . To show that this is indeed the case, in experimental section we take the two notoriously hard nonlinear inverse problems: Fourier phase retrieval and non-uniform deblurring, and show that our method has very strong performance even in such challenging problem settings.



Geometry of DPS and connection to manifold constrained gradient (MCG). Interestingly, our method in the Gaussian measurement case corresponds to the manifold constrained gradient (MCG)step that was proposed in Chung et al. (2022a), when setting $\boldsymbol{W}=\boldsymbol{I}$ from Chung et al. (2022a).


<div style="text-align: center;"><html><body><table border="1"><tr><td rowspan="2">Method</td><td colspan="2">$\mathbf{S R}\left(\times4\right)$</td><td colspan="2">Inpaint (box)</td><td colspan="2">Inpaint (random)</td><td colspan="2">Deblur (gauss)</td><td colspan="2">Deblur (motion)</td></tr><tr><td>FID ↓</td><td>LPIPS↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td></tr><tr><td>DPS (ours)</td><td>39.35</td><td>0.214</td><td>33.12</td><td>0.168</td><td>21.19</td><td>0.212</td><td>44.05</td><td>0.257</td><td>39.92</td><td>0.242</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>62.15</td><td>0.294</td><td>42.93</td><td>0.204</td><td>69.71</td><td>0.587</td><td>74.92</td><td>0.332</td><td>-</td><td>-</td></tr><tr><td>MCG (Chung et al., 2022a)</td><td>87.64</td><td>0.520</td><td>40.11</td><td>0.309</td><td>29.26</td><td>0.286</td><td>101.2</td><td>0.340</td><td>310.5</td><td>0.702</td></tr><tr><td>PnP-ADMM (Chan et al., 2016)</td><td>66.52</td><td>0.353</td><td>151.9</td><td>0.406</td><td>123.6</td><td>0.692</td><td>90.42</td><td>0.441</td><td>89.08</td><td>0.405</td></tr><tr><td>Score-SDE (Song et al., 2021b) (ILVR (Choi et al., 2021))</td><td>96.72</td><td>0.563</td><td>60.06</td><td>0.331</td><td>76.54</td><td>0.612</td><td>109.0</td><td>0.403</td><td>292.2</td><td>0.657</td></tr><tr><td>ADMM-TV</td><td>110.6</td><td>0.428</td><td>68.94</td><td>0.322</td><td>181.5</td><td>0.463</td><td>186.7</td><td>0.507</td><td>152.3</td><td>0.508</td></tr></table></body></html></div>


<div style="text-align: center;">Table 1: Quantitative evaluation (FID, LPIPS) of solving linear inverse problems on FFHQ $256\times256\cdot$ 1k validation dataset. Bold: best, underline: second best. </div>


However, Chung et al. (2022a) additionally performs projection onto the measurement subspace after the update step via (16), which can be thought of as corrections that are made for deviations from perfect data consistency. Borrowing the interpretation of diffusion models from Chung et al.(2022a), we compare the generative procedure geometrically. It was shown that in the context of diffusion models,a single denoising step via sθ corresponds to the orthogonal projection to the data manifold, and the gradient step $\widetilde{\nabla_{\mathbf{x}_{i}}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_{0})\|_{2}^{2}}$ takes a step tangent to the current manifold. For noisy inverse problems, when taking projections on the measurement subspace after every gradient step as in Chung et al. (2022a), the sample may fall off the manifold, accumulate error, and arrive at the wrong solution, as can be seen in Fig. 3a, due to the overly imposing the data consistency that works only for noiseless measurement. On the other hand, our method without the projections on the measurement subspace is free from such drawbacks for noisy measurement (see Fig. 3b).Accordingly, while projections on the measurement subspace are useful for noiseless inverse problems that Chung et al.(2022a) tries to solve,they fail dramatically for noisy inverse problems that we ry to solve. Finally, when used together with the projection steps on the measurement subspace, it was shown that choossing different W for different applications was necessary for MCG, whereas our method is free from such heuristics.



## 4 EXPERIMENTS 

Experimental setup. We test our experiment on two datasets that have diverging characteristic FFHQ 256×256 (Karras et al., 2019), and Imagenet $256\times256$ i (Deng et al., 2009), on 1k validation images each. The pre-trained diffusion model for ImageNet was taken from Dhariwal & Nichol (2021) and was used directly without finetuning for specific tasks. The diffusion model for FFHQ was trained from scratch using 49k training data (to exclude 1k validation set) for 1M steps. All images are normalized to the range [0, 1]. Forward measurement operators are specified as follows:(i) For box-type inpainting, we mask out $128\times128$ box region following Chung et al. (2022a), and for random-type we mask out 92% of the total pixels (all RGB channels). (ii) For super-resolution,bicubic downsampling is performed. (iii) Gaussian blur kernel has size $61\times61$ .with standard deviation of 3.0, and motion blur is randomly generated with the code6, with size $61\times61$ and intensity value 0.5. The kernels are convolved with the ground truth image to produce the measurement. (iv) For phase retrieval, Fourier transform is performed to the image, and only the Fourier magnitude is taken as the measurement. (v) For nonlinear deblurring, we leverage the neural network approximated forward model as in Tran et al. (2021). All Gaussian noise is added to the measurement domain with $\sigma=0.05$ . Poisson noise level is set to $\lambda=1.0$ . More details including the hyper-parameters can be found in Appendix B,D.



We perform comparison with the following methods: Denoising diffusion restoration models (DDRM) (Kawar et al., 2022), manifold constrained gradients (MCG) (Chung et al., 2022a), Plug-and-play alternating direction method of multipliers (PnP-ADMM) (Chan et al.,2016) using DnCNN Zhang et al. (2017) in place of proximal mappings, total-variation (TV) sparsity regularized optimization method (ADMM-TV), and Score-SDE (Song et al., 202ib). Note that Song et al. (2021b) only proposes a method for inpainting, and not for gen


<div style="text-align: center;"><html><body><table border="1"><tr><td>Method</td><td>FID ↓ LPIPS ↓</td></tr><tr><td></td><td>DPS(ours) 55.61 0.399</td></tr><tr><td>OSs</td><td>137.7 0.635</td></tr><tr><td>HIO</td><td>96.40 0.542</td></tr><tr><td>ER</td><td>214.1 0.738</td></tr></table></body></html></div>


<div style="text-align: center;">Table 3: Quantitative evaluation of the Phase Retrieval task (FFHQ). </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_136_122_1087_535.jpg" alt="Image" width="77%" /></div>


<div style="text-align: center;">Figure 4: Results on solving linear inverse problems with Gaussian noise $(\sigma=0.05)$ ).</div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td></td><td colspan="2">SR (×4)</td><td colspan="2">Inpaint (box)</td><td colspan="2">Inpaint (random)</td><td colspan="2">Deblur (gauss)</td><td colspan="2">Deblur (motion)</td></tr><tr><td>Method</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td></tr></thead><tbody><tr><td>DPS (ours)</td><td>50.66</td><td>0.337</td><td>38.82</td><td>0.262</td><td>35.87</td><td>0.303</td><td>62.72</td><td>0.444</td><td>56.08</td><td>0.389</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>59.57</td><td>0.339</td><td>45.95</td><td>0.245</td><td>114.9</td><td>0.665</td><td>63.02</td><td>0.427</td><td>=</td><td>=</td></tr><tr><td>MCG (Chung et al., 2022a)</td><td>144.5</td><td>0.637</td><td>39.74</td><td>0.330</td><td>39.19</td><td>0.414</td><td>95.04</td><td>0.550</td><td>186.9</td><td>0.758</td></tr><tr><td>PnP-ADMM (Chan et al., 2016) Score-SDE (Song et al.,2021b)</td><td>97.27</td><td>0.433</td><td>78.24</td><td>0.367</td><td>114.7</td><td>0.677</td><td>100.6</td><td>0.519</td><td>89.76</td><td>0.483</td></tr><tr><td>(ILVR (Choi et al., 2021))</td><td>170.7</td><td>0.701</td><td>54.07</td><td>0.354</td><td>127.1</td><td>0.659</td><td>120.3</td><td>0.667</td><td>98.25</td><td>0.591</td></tr><tr><td>ADMM-TV</td><td>130.9</td><td>0.523</td><td>87.69</td><td>0.319</td><td>189.3</td><td>0.510</td><td>155.7</td><td>0.588</td><td>138.8</td><td>0.525</td></tr></tbody></table></body></html></div>


<div style="text-align: center;">Table 2: Quantitative evaluation (FID, LPIPS) of solving linear inverse problems on ImageNet $256\times256-$ 1k validation dataset. Bold: best, underline: second best. </div>


eral inverse problems. However, the methodology of iteratively applying projections onto convex sets (POCS) was applied in the same way for super-resolution in iterative latent variable refinement (ILVR) (Choi et al., 2021), and more generally to linear inverse problems in Chung et al. (2022b);thus we simply refer to these methods as score-SDE henceforth.For a fair comparison, we used the same score function for all the different methods that are based on diffusion (i.e. DPs, DDRM, MCG,score-SDE).



For phase retrieval, we compare with three strong baselines that are considered standards: oversampling smoothness (OSS) (Rodriguez et al., 2013), Hybrid input-output (HIO) (Fienup & Dainty,1987), and error reduction (ER) algorithm (Fienup, 1982). For nonlinear deblurring, we compare against the prior arts: blur kernel space (BKS) - styleGAN2 (Tran et al., 2021), based on GAN priors,blur kernel space (BKS) - generic (Tran et al., 2021), based on Hyper-Laplacian priors, and MCG.Further experimental details are provided in Appendix D. For quantitative comparison, we focus on the following two widelyused perceptual metrics - Fréchet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS) distance,with further evaluation with standard metrics: peak signalto-noise-ratio (PSNR), and structural similarity index (SSiM)provided in Appendix E.



Noisy linear inverse problems. We first test our method on diverse linear inverse problems with Gaussian measurement noises. The quantitative results shown in Tables 1,2 illustrate that the proposed method outperforms all the other comparison methods by large margins. Particularly, MCG and Score-SDE (or ILVR) are methods that rely on projections on the measurement subspace, where the generative process is controlled such that the measurement consistency is perfectly met. While this 

<div style="text-align: center;"><img src="imgs/img_in_image_box_741_1157_1113_1474.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;">Figure 5: Results on solving linear inverse problems with Poisson noise,$(\lambda=1.0)$ 1. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_165_112_1061_404.jpg" alt="Image" width="73%" /></div>


<div style="text-align: center;">Figure6: Reultson slvin nolinear invers problems with Gaussian noise $(\sigma=0.05)$ 1.</div>


is useful for noiseless (or negligible noise) problems, in the case where we cannot ignore noise, the solutions overfit to the corrupted measurement (for further discussion, see Appendix C.1). In Fig. 4,we specifically compare our methods with DDRM and PnP-ADMM, which are two methods that are known to be robust to measurement noise. Our method is able to provide high-quality reconstructions that are crisp and realistic on all tasks. On the other hand, we see that DDRM performs poorly on image inpainting tasks where the dimensionality of the measurements are very low, and tend to produce blurrier results on both SR, and deblurring tasks. We further note that DDRM relies on SVD,and hence is only able to solve problems where the forward measurement matrix can be efficiently implemented (e.g. separable kernel in the case of deblurring). Hence, while one can solve Gaussian deblurring, one cannot solve problems such as motion deblur, where the point spread function (PSF)is much more complex. Contrarily, our method is not restricted by such conditions, and can be always used regardless of the complexity. The results of the Poisson noisy linear inverse problems are presented in Fig.5.Consistent with the Gaussian case, DPS is capable of producing high quality reconstructions that closely mimic the ground truth. From the experiments, we further observe that the weighted least squares method adopted in Algorithm 2 works best compared to other choices that can be made for Poisson inverse problems (for further analysis, see Appendix C.4).

Nonlinear inverse problems. We show the quantitative results of phase retrieval in Table 3,and the results of nonlinear deblurring in Table 4. Representative results are illustrated in Fig. 6.



accurate reconstruction for the given phase retrieval problem, capturing most of the high frequency details. However, we also observe that we do not always get high quality reconstructions. In fact, due to the non-uniqueness of the phase-retrieval under some conditions, widely used methods such as HIO are also dependent on the initializations (Fienup, 1978), and hence it is considered standard practice to first generate multiple reconstructions, and take the best sample. Following this, when reporting our quantitative metrics, we generate 4 different samples for all the methods,and report the metric based on the best samples. We see that DPS 


<div style="text-align: center;"><html><body><table border="1"><tbody><tr><td>Method</td><td>FID ↓ LPIPS ↓</td></tr><tr><td>DPS(ours)</td><td>41.86 0.278</td></tr><tr><td>BKS-styleGAN2 63.18 0.407</td><td></td></tr><tr><td>BKS-generic</td><td>141.0 0.640</td></tr><tr><td>MCG</td><td>180.1 0.695</td></tr></tbody></table></body></html></div>


<div style="text-align: center;">Table 4: Quantitative evaluation of the non-uniform deblurring task (FFHQ). </div>


outperforms other methods by a large margin. For the case of nonlinear deblurring, we again see that our method performs the best, producing highly realistic samples. BKS-styleGAN2 (Tran et al.,2021) leverages GAN prior and hence generates feasible human faces, but heavily distorts the identity.BKS-generic utilizes the Hyper-Laplacian prior (Krishnan & Fergus, 2009), but is unable to remove artifacts and noise properly. MCG amplifies noise in a similar way that was discussed in Fig. 7.

## 5 CONCLUSION 

In this paper, we proposed diffusion posterior sampling (DPS) strategy for solving general noisy (both signal dependent/independent) inverse problems in imaging. Our method is versatile in that it can also be used for highly noisy and nonlinear inverse problems. Extensive experiments show that the proposed method outperforms existing state-of-the-art by large margins, and also covers the widest range of problems.



## ACKNOWLEDGMENTS 

This work was supported by the National Research Foundation of Korea under Grant NRF2020R1A2B5B03001980, by the Korea Medical Device Development Fund grant funded by the Korea government (the Ministry of Science and ICT, the Ministry of Trade, Industry and Energy, the Ministry of Health & Welfare, the Ministry of Food and Drug Safety) (Project Number: 1711137899,KMDF_PR_20200901_0015), and by the KAIST Key Research Institute (Interdisciplinary Research Group) Project.



## REFERENCES 

Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.
Thilo Balke, Fernando Davis, Cristina Garcia-Cardona, Michael McCann, Luke Pfister, and Brendt Wohlberg. Scientific Computational Imaging COde (SCICO). Software library available from https://github.com/lanl/scico,2022.
Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6228–6237, 2018.Yu M Bruck and LG Sodin. On the ambiguity of the image reconstruction problem. Optics communications, 30(3):304–308, 1979.
Stanley H Chan, Xiran Wang, and Omar A Elgendy. Plug-and-play admm for image restoration:Fixed-point convergence and applications. IEEE Transactions on Computational Imaging, 3(1):84–98, 2016.
Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. ILVR:Conditioning method for denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.
Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated MRI. Medical Image Analysis, pp. 102479, 2022.
Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. arXiv preprint arXiv:2206.00941, 2022a.Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022b.Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition,pp. 248–255. Ieee, 2009.
Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis.In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021.
Bradley Efron. Tweedie's formula and selection bias. Journal of the American Statistical Association,106(496):1602–1614, 2011.
C Fienup and J Dainty. Phase retrieval and image reconstruction for astronomy. Image recovery:theory and application, 231:275, 1987.
James R Fienup. Reconstruction of an object from the modulus of its fourier transform. Optics letters,
3(1):27–29, 1978.
James R Fienup. Phase retrieval algorithms: a comparison. Applied optics, 21(15):2758–2769, 1982.Xiang Gao, Meera itharam, and Adian ERoitberg. Bounds on the jensen gap,and imlications for mean-concentrated distributions. arXiv preprint arXiv:1712.05267, 2017.

MHMH Hayes. The reconstruction of a multidimensional sequence from the phase or magnitude of its fourier transform. IEEE Transactions on Acoustics, Speech, and Signal Processing, 30(2):140–154, 1982.
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.WM Hubbard. The approximation of a poisson distribution by a gaussian distribution. Proceedings of the IEEE, 58(9):1374–1375, 1970.
Ajil Jalal, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G Dimakis, and Jon Tamir. Robust Compressed Sensing MRI with Deep Generative Priors. In Advances in Neural Information Processing Systems, volume 34, pp. 14938–14954, 2021.
Zahra Kadkhodaie and Eero Simoncelli. Stochastic solutions for linear inverse problems using the prior implicit in a denoiser. In Advances in Neural Information Processing Systems,volume 34, pp.13242–13254. Curran Associates, Inc., 2021.
Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.
Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. Advances in Neural Information Processing Systems, 34:21757–21769, 2021.Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In ICLR Workshop on Deep Generative Models for Highly Structured Data, 2022.Kwanyoung Kim and Jong Chul Ye. Noise2score: Tweedie's approach to self-supervised image denoising without clean images. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https:/ /openreview.net/forum?id=ZqEUs3sTRUo.
Robert Hildreth Kingston. Detection of optical and infrared radiation, volume 10. Springer, 2013.Dilip Krishnan and Rob Fergus. Fast image deconvolution using hyper-laplacian priors. Advances in neural information processing systems, 22, 209.
Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=PlkwVd2yBkY.
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. arXiv preprint arXiv:2206.00927,2022.
Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3883–3891, 2017.
Herbert E Robbins. An empirical bayes approach to statistics. In Breakthroughs in statistics, pp.388–394. Springer, 1992.
Jose A Rodriguez, Rui Xu, C-C Chen, Yunfei Zou, and Jianwei Miao. Oversampling smoothness:an effective algorithm for phase retrieval of noisy diffraction intensities. Journal of applied crystallography, 46(2):312–318, 2013.
Slavko Simic. On a global upper bound for jensen's inequality. Journal of mathematical analysis and applications, 343(1):414–419, 2008.
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR, 2021a.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR, 2021b.Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations,2022.
Charles M Stein. Estimation of the mean of a multivariate normal distribution. The annals of Statistics,pp. 1135–1151, 1981.
Phong Tran, Anh Tuan Tran, Quynh Phung, and Minh Hoai. Explore image deblurring via encoded blur kernel space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.11956–11965, 2021.
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.
Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser:Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017.


## A PROOFS 

Lemma 1 (Tweedie's formula). Let $p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}})$  belong to the exponential family distribution 

$$p(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}})=p_{0}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}})\operatorname{e x p}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}}}^{\top}T(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}})-\varphi(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}}}})),$$

where η is thecanonical vectorof the family,$T(\boldsymbol{y})$ is some function of $y$ , and $\varphi(\pmb\eta)$ is the cumulant generation function which normalizes the density, and $p_{0}(\pmb{y})$ is the density up to the scale factor when $\mathbf{\eta}=\mathbf{0}$ . Then, the posterior mean $\hat{\pmb{\eta}}:=\mathbb{E}[\pmb{\eta}|\pmb{y}]$ should satisfy [Ta]

$$(\nabla_{\pmb{y}}T(\pmb{y}))^{\top}\hat{\pmb{\eta}}=\nabla_{\pmb{y}}\operatorname{l o g}p(\pmb{y})-\nabla_{\pmb{y}}\operatorname{l o g}p_{0}(\pmb{y})$$

Proof. Marginal distribution $p(\pmb{y})$ could be expressed as 

$$\begin{aligned}{p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}})}&{{}=\int p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}})p(\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}})d\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}}\\ {}&{{}=\int p_{0}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}})\operatorname{e x p}\big(\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}^{\top}T(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}})-\varphi(\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}})\big)p(\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}})d\mathbf{\mathbf{\mathbf{\mathbf{\eta}}}}.}\\ \end{aligned}$$

Then,the iio iutio $p(\pmb{y})$ with respect to y becomes 

$$\begin{aligned}{\nabla_{\pmb{y}}p(\pmb{y})}&{{}=\nabla_{y}p_{0}(\pmb{y})\int\operatorname{e x p}\big(\pmb{\eta}^{\top}T(\pmb{y})-\varphi(\pmb{\eta})\big)p(\pmb{\eta})d\pmb{\eta}+\int(\nabla_{\pmb{y}}T(\pmb{y}))^{\top}\pmb{\eta}p_{0}(\pmb{y})\operatorname{e x p}\big(\pmb{\eta}^{\top}T(\pmb{y})-\varphi(\pmb{\eta})\big)p(\pmb{\eta})d\pmb{\eta}}\\ {}&{{}=\frac{\nabla_{y}p_{0}(\pmb{y})}{p_{0}(\pmb{y})}\int p(\pmb{y}|\pmb{\eta})p(\pmb{\eta})d\pmb{\eta}+(\nabla_{\pmb{y}}T(\pmb{y}))^{\top}\int\pmb{\eta}p(\pmb{y}|\pmb{\eta})p(\pmb{\eta})d\pmb{\eta}}\\ {}&{{}=\frac{\nabla_{y}p_{0}(\pmb{y})}{p_{0}(\pmb{y})}p(\pmb{y})+(\nabla_{\pmb{y}}T(\pmb{y}))^{\top}\int\pmb{\eta}p(\pmb{y},\pmb{\eta})d\pmb{\eta}}\\ \end{aligned}$$

Therefore,

$$\frac{\nabla_{y}p(\mathbf{y})}{p(\mathbf{y})}=\frac{\nabla_{y}p_{0}(\mathbf{y})}{p_{0}(\mathbf{y})}+(\nabla_{\mathbf{y}}T(\mathbf{y}))^{\top}\int\mathbf{\eta}p(\mathbf{\eta}|\mathbf{y})d\mathbf{\eta}.$$

which is equivalent to 

$$(\nabla_{\mathbf{y}}T(\mathbf{y}))^{\top}\mathbb{E}[\mathbf{\eta}|\mathbf{y}]=\nabla_{\mathbf{y}}\operatorname{l o g}p(\mathbf{y})-\nabla_{\mathbf{y}}\operatorname{l o g}p_{0}(\mathbf{y})$$

This concludes the proof.

Proposition 1. For the case of VP-SDE or DDPM sampling,$p(\mathbf{{{x}}}_{0}|\mathbf{{{x}}}_{t})$ has the unique posterior mean at 



$$\hat{\pmb x}_{0}:=\mathbb{E}[{\pmb x}_{0}|{\pmb x}_{t}]=\frac{1}{\sqrt{\bar{\alpha}(t)}}({\pmb x}_{t}+(1-\bar{\alpha}(t))\nabla_{{\pmb x}_{t}}\operatorname{l o g}p_{t}({\pmb x}_{t}))$$

Proof. For the case of VP-SDE and DDPM forward sampling in (8), we have 

$$p(\mathbf{\mathit{x}}_{t}|\mathbf{\mathit{x}}_{0})=\frac{1}{(2\pi(1-\bar{\alpha}(t)))^{d/2}}\operatorname{e x p}\left(-\frac{\|\mathbf{\mathit{x}}_{t}-\sqrt{\bar{\alpha}(t)}\mathbf{\mathit{x}}_{0}\|^{2}}{2(1-\bar{\alpha}(t))}\right),$$

which is a Gaussian distribution. The corresponding canonical decomposition is then given by 

$$p(\mathbf{\mathbf{\mathit{x}}}_{t}|\mathbf{\mathbf{\mathit{x}}}_{0})=p_{0}(\mathbf{\mathbf{\mathit{x}}}_{t})\operatorname{e x p}\left(\mathbf{\mathbf{\mathit{x}}}_{0}^{\top}T(\mathbf{\mathbf{\mathit{x}}}_{t})-\varphi(\mathbf{\mathbf{\mathit{x}}}_{0})\right),$$

where 

$$\begin{aligned}{p_{0}(\mathbf{{\mathbf{\mathit{x}}}}_{t})}&{{}:=\frac{1}{(2\pi(1-\bar{\alpha}(t)))^{d/2}}\operatorname{e x p}\left(-\frac{\|\mathbf{{\mathbf{\mathit{x}}}}_{t}\|^{2}}{2(1-\bar{\alpha}(t))}\right)}\\ {T(\mathbf{{\mathbf{\mathit{x}}}}_{t})}&{{}:=\frac{\sqrt{\bar{\alpha}(t)}}{1-\bar{\alpha}(t)}\mathbf{{\mathbf{\mathit{x}}}}_{t}}\\ {\varphi(\mathbf{{\mathbf{\mathit{x}}}}_{0})}&{{}:=\frac{\bar{\alpha}(t)\|\mathbf{{\mathbf{\mathit{x}}}}_{0}\|^{2}}{2(1-\bar{\alpha}(t))}}\\ \end{aligned}$$

Therefore, using (24), we have 

$$\frac{\sqrt{\bar{\alpha}(t)}}{1-\bar{\alpha}(t)}\hat{\mathbf{{x}}}_{0}=\nabla_{\mathbf{{x}}_{t}}\operatorname{l o g}p_{t}(\mathbf{{x}}_{t})+\frac{1}{1-\bar{\alpha}(t)}\mathbf{{x}}_{t}$$

which leads to 

$$\hat{\mathbf{\mathit{x}}}_{0}=\frac{1}{\sqrt{\bar{\alpha}(t)}}\left(\mathbf{\mathit{x}}_{t}+(1-\bar{\alpha}(t))\nabla_{\mathbf{\mathit{x}}_{t}}\operatorname{l o g}p_{t}(\mathbf{\mathit{x}}_{t})\right)$$

This concludes the proof.

□

Proposition2 (Jensen gap upperbound (Gao et al., 2017)). Define the absolute cenetered moment as $m_{p}:=\sqrt[p]{\mathbb{E}[|X-\mu|^{p}]}$ , and the mean as $\mu=\mathbb{E}[X]$ . Assume that for $\alpha>0,$ , there exists a positive number K such that for any $x\in\mathbb{R},|f(x)-f(\mu)|\leq K|x-\mu|^{\alpha}$ . Then,

$$\begin{aligned}{|\mathbb{E}[f(X)-f(\mathbb{E}[X])]|}&{{}\leq\int|f(X)-f(\mu)|d p(X)}\\ {}&{{}\leq K\int|x-\mu|^{\alpha}d p(X)\leq M m_{\alpha}^{\alpha}.}\\ \end{aligned}$$

Lemma 2.  Let $\phi(\cdot)$ be a univariate Gaussian density function with mean  and variance $\sigma^{2}$ . There exists a constant L such that ∀x,$y\in\mathbb{R}$ 



$$|\phi(x)-\phi(y)|\leq L|x-y|,$$

where $\begin{array}{r}{L=\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp{(-\frac{1}{2\sigma^{2}})}.}\end{array}$ 

Proof. As $\phi^{\prime}$ is continuous and bounded, we use the mean value theorem to get 

$$\forall(x,y)\in\mathbb{R}^{2},\:|\phi(x)-\phi(y)|\leq\|\phi^{\prime}\|_{\infty}|x-y|.$$

Since L is the minimal value for (34), we have that $L\leq\|\phi^{\prime}\|_{\infty}$ . Taking the limit $y\rightarrow x{\mathrm{~\bf{g i v e s}}}$ ：$|\phi^{\prime}(x)|\leq L$ , and thus $\|\phi^{\prime}\|_{\infty}\leq L$ . Hence 



$$L=\|\phi^{\prime}\|_{\infty}=\|-\frac{x-\mu}{\sigma^{2}}\phi(x)\|_{\infty}.$$

Since the derivative of $\phi^{\prime}$ is given as 

$$\phi^{\prime\prime}(x)=\sigma^{-2}(1-\sigma^{-2}(x-\mu)^{2})\phi(x),$$

and the maximum is attained when $x=1\pm\sigma^{2}\mu$ . , we have 

$$L=\|\phi^{\prime}\|_{\infty}=\frac{e^{-1/2\sigma^{2}}}{\sqrt{2\pi\sigma^{2}}}.$$

Lemma 3. Let $\phi(\cdot)$ baG $\sigma^{2}\pmb{I}$ . There exists a constant L such that ∀x, $\pmb{y}\in\mathbb{R}^{d}$ 



$$\|\phi({\bf x})-\phi({\bf y})\|\leq L\|{\bf x}-{\bf y}\|,$$

where $\begin{array}{r}{L=\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}}\end{array}$ 

Proof.

$$\begin{aligned}{\|\phi(\mathbf{\mathbf{\mathbf{\phi}}})-\phi(\mathbf{\mathbf{\mathbf{\mathbf{y}}}})\|}&{{}\leq\operatorname*{m a x}_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{z}}}}}}\|\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{z}}}}}}}\phi(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{z}}}}}}})\|\cdot\|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}-\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}\|}\\ {}&{{}=\underbrace{\frac{d}{\sqrt{2\pi\sigma^{2}}}\operatorname{e x p}\left(-\frac{1}{2\sigma^{2}}\right)}_{L}\cdot\|\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\phi}}}}}}}-\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}}\|}\\ \end{aligned}$$

where the second inequality comes from that each element  of $\nabla_{\pmb{z}}\phi(\pmb{z})$ is bounded by $\begin{array}{r}{\frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\left(-\frac{1}{2\sigma^{2}}\right)}\end{array}$ □

Theorem 1. For the given measurement model (6) with $\mathbf{\mathit{n}}\sim\mathcal{N}(0,\sigma^{2}\mathbf{\mathit{I}})$ ,we have 

$$p(\mathbf{\mathbf{\mathit{y}}}|\mathbf{\mathbf{\mathit{x}}}_{t})\simeq p(\mathbf{\mathbf{\mathit{y}}}|\hat{\mathbf{\mathbf{\mathit{x}}}}_{0}),$$

where the approximation error can be quantified with the Jensen gap, which is upper bounded by 

$$\mathcal{J}\leq\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|m_{1},$$

$$where \textstyle\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|:=\operatorname*{m a x}_{\pmb{x}}\|\nabla_{\pmb{x}}\mathcal{A}(\pmb{x})\|\:{a n d}\:m_{1}:=\int\|\pmb{x}_{0}-\hat{\pmb{x}}_{0}\|p(\pmb{x}_{0}|\pmb{x}_{t})\:d\pmb{x}_{0}.$$

Proof.

$$\begin{aligned}{p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})}&{{}=\int p(\mathbf{\mathbf{\mathbf{\mathit{y}}}}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0})p(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})d\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}}\\ {}&{{}=\mathbb{E}_{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0}\sim p(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0}|\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{t})}[f(\mathbf{\mathbf{\mathbf{\mathit{x}}}}_{0})]}\\ \end{aligned}$$

Here,$f(\cdot):=h(\mathcal{A}(\cdot))$ where $\mathcal{A}$ is the forward operator and $h({\pmb x})$ is the multivariate normal distribution with mean y and the covariance $\sigma^{2}\pmb{I}$ . Therefore, we have 

$$\begin{aligned}{J(f,p(\mathbf{x}_{0}|\mathbf{x}_{t}))}&{{}=|\mathbb{E}[f(\mathbf{x}_{0})]-f(\mathbb{E}[\mathbf{x}_{0}])|=|\mathbb{E}[f(\mathbf{x}_{0})]-f(\hat{\mathbf{x}}_{0})|}\\ {}&{{}=|\mathbb{E}[h(\mathcal{A}(\mathbf{x}_{0}))]-h(\mathcal{A}(\hat{\mathbf{x}}_{0}))|}\\ {}&{{}\leq\int|h(\mathcal{A}(\mathbf{x}_{0}))-h(\mathcal{A}(\hat{\mathbf{x}}_{0}))|d P(\mathbf{x}_{0}|\mathbf{x}_{t})}\\ {}&{{}\overset{(\mathfrak{b})}{\leq}\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}\int\|\mathcal{A}(\mathbf{x}_{0})-\mathcal{A}(\hat{\mathbf{x}}_{0})\|d P(\mathbf{x}_{0}|\mathbf{x}_{t})}\\ {}&{{}\overset{(\mathfrak{c})}{\leq}\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}\|\nabla_{\mathbf{x}}\mathcal{A}(\mathbf{x})\|\int\|\mathbf{x}_{0}-\hat{\mathbf{x}}_{0}\|d P(\mathbf{x}_{0}|\mathbf{x}_{t})}\\ {}&{{}\overset{(\mathfrak{d})}{\leq}\frac{d}{\sqrt{2\pi\sigma^{2}}}e^{-1/2\sigma^{2}}\|\nabla_{\mathbf{x}}\mathcal{A}(\mathbf{x})\|m_{1}}\\ \end{aligned}$$

where $d P(\mathbf{\mathit{x}}_{0}|\mathbf{\mathit{x}}_{t})=p(\mathbf{\mathit{x}}_{0}|\mathbf{\mathit{x}}_{t})\mathop{d\mathbf{\mathit{x}}_{0}}$ 1, (b) is the result of Lemma 3, (c) is from the intermediate value theorem, and (d) is from Proposition 2.



## B INVERSE PROBLEM SETUP 

Super-resolution. The forward model for super-resolution is defined as 

$$\begin{aligned}{\mathbf{\mathit{y}}\sim\mathcal{N}(\mathbf{\mathit{y}}|\mathbf{\mathit{L}}^{f}\mathbf{\mathit{x}},\sigma^{2}\mathbf{\mathit{I}}),\qquad}&{{}\operatorname{(G a u s s i a n)}}\\ {\mathbf{\mathit{y}}\sim\mathcal{P}(\mathbf{\mathit{y}}|\mathbf{\mathit{L}}^{f}\mathbf{\mathit{x}};\lambda),\qquad}&{{}\operatorname{(P o i s s o n)}}\\ \end{aligned}$$

where $\pmb{L}^{f}\in\mathbb{R}^{n\times d}$ r $f,$ and $\mathcal{P}$ denotes the Poisson distribution with the parameter λ.



Inpainting. For both box-type and random-type inpainting, the forward model reads 

$$\begin{aligned}{\mathbf{y}\sim\mathcal{N}(\mathbf{y}|\mathbf{P}\mathbf{x},\sigma^{2}\mathbf{I}),\qquad}&{{}\operatorname{(G a u s s i a n)}}\\ {\mathbf{y}\sim\mathcal{P}(\mathbf{y}|\mathbf{P}\mathbf{x};\lambda),\qquad}&{{}\operatorname{(P o i s s o n)}}\\ \end{aligned}$$

where $\pmb{P}\in\{0,1\}^{n\times d}$ is the masking matrix that consists of elementary unit vectors.

Linear Deblurring. For both Gaussian and motion deblurring, the measurement model is given as 

$$\begin{aligned}{\mathbf{\mathit{y}}\sim\mathcal{N}(\mathbf{\mathit{y}}|\mathbf{\mathit{C}}^{\psi}\mathbf{\mathit{x}},\sigma^{2}\mathbf{\mathit{I}}),\qquad}&{{}(\operatorname{G a u s s i a n})}\\ {\mathbf{\mathit{y}}\sim\mathcal{P}(\mathbf{\mathit{y}}|\mathbf{\mathit{C}}^{\psi}\mathbf{\mathit{x}};\lambda),\qquad}&{{}(\operatorname{P o i s s o n})}\\ \end{aligned}$$

where $\mathbf{C}^{\psi}\in\mathbb{R}^{n\times d}$ is the block Hankl matrix that efctivl induces convolution with the given blur kernel $\psi$ ,

<div style="text-align: center;"><img src="imgs/img_in_image_box_115_112_1064_555.jpg" alt="Image" width="77%" /></div>


<div style="text-align: center;">Figure 7: Failure cases of MCG (Chung et al., 2022a) on noisy inverse problems due to noise amplification. </div>


Nonlinear deblurring. We leverage the nonlinear blurring process that was proposed in the GOPRO dataset (Nah et al., 2017), where the blurring process is not defined as a convolution, but rather as an integration of sharp images through the time frame. Specifically, in the discrete sense the measurement model reads 



$$\pmb{y}=\int b\left(\frac{1}{M}\sum_{i=1}^{M}\pmb{x}[i]\right),\quad i=1,\ldots,T,$$

where $b(\mathbf{\mathit{x}})=\mathbf{\mathit{x}}^{1/2.2}$ is the nonlinear camera response function, and $T$ denotes the total time frames.While we could directly use (56) as our forward model, note that this is only possible when we have multiple sharp time frames at hand (e.g. when leveraging GOPRO dataset directly). Recently, there was an effort to distill the forward model through a neural network (Tran et al., 2021). Particularly,when we have a set of blurry-sharp image pairs $\{(\mathbf{{{x}}}_{i},\mathbf{{{y}}}_{i})\}_{i=1}^{N}$ , one can train a neural network to estimate the forward model as 



$$\phi^{*}=\mathop{\operatorname{a r g}\operatorname*{m i n}}_{\theta}\sum_{i=1}^{N}\|\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}_{i}-\mathcal{F}_{\phi}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{i},\mathcal{G}_{\phi}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{i},\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}_{i}))\|,$$

where $\mathcal{G}_{\phi}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{i},\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}}_{i})$ extract $\mathcal{F}_{\phi}$ takes in ${\boldsymbol{x}}_{i}$ $\mathcal{G}_{\phi}(\pmb{x}_{i},\pmb{y}_{i})$ to generate the blurry image. When using $\mathcal{F}_{\phi}$ at deployment to generate new synthetic blurry images,one can simply replace $\widetilde{\mathcal{G}_{\phi}}(\mathbf{\mathit{x}}_{i},\mathbf{\mathit{y}}_{i})$ with a Gaussian random vector k. Consequently, our forward model reads 



$$\begin{aligned}{\mathbf{\mathit{y}}\sim\mathcal{N}(\mathbf{\mathit{y}}|\mathcal{F}_{\phi}(\mathbf{\mathit{x}},\mathbf{\mathit{k}}),\sigma^{2}\mathbf{\mathit{I}}),}&{{}\mathbf{\mathit{k}}\in\mathbb{R}^{k},\mathbf{\mathit{k}}\in\mathcal{N}(0,\sigma_{k}^{2}\mathbf{\mathit{I}}),\qquad\operatorname{(G a u s s i a n)}}\\ {\mathbf{\mathit{y}}\sim\mathcal{P}(\mathbf{\mathit{y}}|\mathcal{F}_{\phi}(\mathbf{\mathit{x}},\mathbf{\mathit{k}});\lambda),}&{{}\mathbf{\mathit{k}}\in\mathbb{R}^{k},\mathbf{\mathit{k}}\in\mathcal{N}(0,\sigma_{k}^{2}\mathbf{\mathit{I}}),\qquad\operatorname{(P o i s s o n)}}\\ \end{aligned}$$

where  is the dimensionality ofthe latnt ector k, and $\sigma_{k}^{2}$ is the variance of the vector.

Phase Retrieval. The forward measurement model is usually given as 

$$\begin{aligned}{\mathbf{\mathit{\mathbf{y}}}\sim\mathcal{N}(\mathbf{\mathit{\mathbf{y}}}||\mathbf{\mathit{\mathbf{F}}}\mathbf{\mathit{\mathbf{x}}}_{0}|,\sigma^{2}\mathbf{\mathit{\mathbf{I}}}),\qquad}&{{}(\operatorname{G a u s s i a n})}\\ {\mathbf{\mathit{\mathbf{y}}}\sim\mathcal{P}(\mathbf{\mathit{\mathbf{y}}}||\mathbf{\mathit{\mathbf{F}}}\mathbf{\mathit{\mathbf{x}}}_{0}|;\lambda),\qquad}&{{}(\operatorname{P o i s s o n})}\\ \end{aligned}$$

where F denotes the 2D Discrete Fourier Trasform (DFT) matrix. In another words, the phase of the Fourier measurements are nulled, and our aim is to impute the missing phase information. As the problem is highly ill-posed, one typically incorporates the oversampling in order to induce the 

$$\zeta^{\prime}=0.001$$

<div style="text-align: center;">Figure 8: Effect of step size $\zeta^{\prime}$ on the results </div>


uniqueness condition (Hayes, 1982; Bruck & Sodin, 1979), usually specified as 

$$\begin{array}{r c l}{\mathbf{\mathit{\mathbf{y}}}\sim\mathcal{N}(\mathbf{\mathit{\mathbf{y}}}||\mathbf{\mathit{\mathbf{F}}}\mathbf{\mathit{\mathbf{P}}}\mathbf{\mathit{\mathbf{x}}}_{0}|,\sigma^{2}\mathbf{\mathit{\mathbf{I}}}),}&{}&{(\operatorname{G a u s s i a n})}\\ {\mathbf{\mathit{\mathbf{y}}}\sim\mathcal{P}(\mathbf{\mathit{\mathbf{y}}}||\mathbf{\mathit{\mathbf{F}}}\mathbf{\mathit{\mathbf{P}}}\mathbf{\mathit{\mathbf{x}}}_{0}|;\lambda),}&{}&{(\operatorname{P o i s s o n})}\\ \end{array}$$

where P denotes the oversampling matrix with ratio $k/n$ 

Poisson noise simulation. To simulate the Poisson noise, we assume that each measurement pixel is a source of photon, where the number of photons is proportional to the discrete pixel value between 0and 255. Thus, we sample noisy measurement values from the Poisson distribution with the mean value of the clean measurement values. Here, the clean measurement is $\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{0})$ ), which is an image after applying the forward operation. Then, we clip the values by [0, 255] and normalize to [-1, 1].

## C ABLATION STUDIES AND DISCUSSION 

### C.1 NOISE AMPLIFICATION BY PROJECTION 

As discussed in the experiments, methods that rely on projections fail dramatically when solving inverse problems with excessive amount of noise in the measurement. Even worse, for many problems such as SR or deblurring, noise gets amplified during the projection step due to the operator transpose $\pmb{A}^{T}$ being applied. This downside is clearly depicted in Fig. 7, where we show the failure cases of MCG (Chung et al., 2022a) on noisy super-resolution. In contrast, our method does not rely on such projections, and thus is much more robust to the corrupted measurements. Notably, we find that MCG also fails dramatically in SR even when there is no noise existent, while it performs well on some of the other tasks (e.g. inpainting). We can conclude that the proposed method works generally well across a broader range of inverse problems, whether or not there is noise in the measurement.

### C.2 EFFECT OF STEP SIZE $\zeta^{\prime}$ 

There is one hyper-parameter in our DPS solver, and that is the step size. As this value is essentially the weight that is given to the likelihood (i.e. data consistency) of the inverse problem, we can expect that the values being too high or too low will cause problems. In Fig. 8, we show the trend of the reconstruction results when varying the se sie $\zeta_{i}$ . Note that we instead use the notation $\zeta^{\prime}~{\triangleq}~\zeta_{i}\|{\mathbf{\mathit{y}}}-{\mathcal{A}}(\hat{{\mathbf{\mathit{x}}}}_{0}({\mathbf{\mathit{x}}}_{i}))\|$ for brevity. Here, we see that with low values of $\zeta^{\prime}<0.1$ , we achieve results that are not consistent with the given measurement. On the other hand, when we crank up the values too high $(\zeta^{\prime}>5)$ ), we observe saturation arfiacts that tend to amplify the noise. From our experiments, weconclude that it is best practice to set the $\zeta^{\prime}$ values in the range [0.1, 1.0] for best results. Specific values for all the experiments are presented in Appendix D.

### C.3 OTHER STEP SIZE SCHEDULES 

While the proposed step size schedule in C.2 yields good results, there can be many other choices that one can take.In this section, we conduct an ablation study to compare against other choices.Specifically, we test 100 images for Gaussian deblurring (Gaussian noise,$\sigma=0.05)$ ) on FFHQ, and 

<div style="text-align: center;"><img src="imgs/img_in_image_box_127_125_1083_496.jpg" alt="Image" width="78%" /></div>


<div style="text-align: center;">Figure 9: Ablation study on the choice of step size schedule for DPS. (a) Measurement,$(\mathbf{b}\mathbf{-c})$ exponential decay with initial values 0.3, 1.0, (d-e) linear decay with initial values 0.3, 1.0, (f)$\propto\bar{1}/\sigma^{2}$ (g) ours, (h) ground truth. </div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td>Strategy</td><td colspan="2">Constant</td><td colspan="2">Linear decay</td><td colspan="2">Exponential decay</td></tr></thead><tbody><tr><td>Initial value</td><td>1.0 (Ours) $1/\sigma^{2}$</td><td>0.3 1.0</td><td>0.3 1.0</td><td></td><td></td><td></td></tr><tr><td>LPIPS ↓ $0.247\pm0.045$ $0.727\pm0.038$</td><td>LPIPS ↓</td><td>$\underline{{0.251}}\pm0.044$ $0.287\pm0.045$</td><td>0.442 ± 0.108 $0.421\pm0.065$</td><td></td><td></td><td></td></tr></tbody></table></body></html></div>


<div style="text-align: center;">Table 5: Ablation study on step size scheduling. Bold: best, underline: second best.</div>


compute the average perceptual distance (LPIPS) against the ground truth. We compare against the following three choices: 1) Linearly decaying steps $\begin{array}{l}{\zeta_{i}^{\prime}=\zeta_{r m i n i t}^{\prime}\times{\left({1-\frac{i}{N}}\right)},2)}\\ \end{array}$ )exponentially decaying steps $\zeta_{i}^{\prime}=\zeta_{\mathrm{i n i t}}^{\prime}\times\gamma^{i}$ ,with $\gamma=0.99,3)$ ) directly using step size proportional to $1/\sigma^{2}$ as in eq. 16.



We present qualitative analysis in Fig. 9. From the figure, it is clear that the proposed schedule produces the best result that most closely matches the ground truth in terms of perception. For decaying step sizes, we often yield results that are coarsely similar to the ground truth, but varies in the fine details, as the information about the measurement is less incorporated in the later steps of the diffusion. From Fig. 9, we see that taking step sizes proportional to $1/\sigma^{2}$ ,motivated by direct derivation from the gaussian forward model, yields poor results. We see similar results with the quantitative metrics presented in Table. 5.



### C.4 POISSON INVERSE PROBLEMS 

For inverse problems corrupted with Poisson noise, more care needs to be taken compared to the Gaussian noise counterparts, as the noise is signal-dependent and therefore harder to account for. In this section, we discuss the different choices of likelihood functions that can be made, and clarify the choice (20) used in all our experiments. One straightforward option is to directly use the Poisson likelihood model without the Gaussian approximation. From (17), we have that 

$$\begin{aligned}{\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})}&{{}=\sum_{j=1}^{n}\operatorname{l o g}[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}-[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}-\operatorname{l o g}(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}_{j}!)}\\ {\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})}&{{}=-\alpha\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}}_{t}}\left[\sum_{j=1}^{n}\operatorname{l o g}[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}-[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{0})]_{j}\right],}\\ \end{aligned}$$

<div style="text-align: center;"><img src="imgs/img_in_image_box_124_121_1115_783.jpg" alt="Image" width="80%" /></div>


<div style="text-align: center;">Figure 10: Differences in the reconstruction results when using different choices for imposing data consistency for Poisson linear inverse problems. </div>


which we refer to as Poisson-direct. Moreover, one can use the Gaussian approximated version of the Poisson measurement model given in (18)



$$\begin{aligned}{\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})}&{{}=\sum_{j=1}^{n}-\frac{1}{2}\operatorname{l o g}\left[2\pi[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}\right]-\frac{(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}_{j}-[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j})^{2}}{2[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}}}\\ {\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\operatorname{l o g}p(\mathbf{\mathbf{\mathbf{\mathbf{y}}}}|\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})}&{{}=\alpha\nabla_{\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{x}}}}}_{t}}\left[\sum_{j=1}^{n}\frac{1}{2}\operatorname{l o g}\left[2\pi[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}\right]+\frac{(\mathbf{\mathbf{\mathbf{\mathbf{\mathbf{y}}}}}_{j}-[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j})^{2}}{2[\mathcal{A}(\mathbf{\mathbf{\mathbf{\mathbf{x}}}}_{0})]_{j}}\right],}\\ \end{aligned}$$

which we refer to as Poisson-Gaussian. Next, we can use our choice in (19) to arrive at (20), which is the proposed method. Finally, while irrelevant with the noise model, we can also still use the same least sqaures (LS) method used for Gaussian noise (we refer to this method as Poisson-LS), as due to the central limit theorem, Poisson noise is nearly Gaussian in the high SNR level regime. In Fig. 10, we show representative results achieved by using each choice. From the experiments, we observe that Poisson-direct is unstable due to the log term in the likelihood, hence often diverging.We also observe that the residual $\pmb{y}-\mathcal{A}(\hat{\pmb{x}}_{0})$ fails to converge, hinting that the information from the measurement is not effectively integrated into the generative process. For Poisson-Gaussian, we see that the weighting term of the MSE is problematic, and this term prevents the process from proper convergence. Both the proposed method and Poisson-LS are stable, but Poisson-LS tends to blur out the relevant details from the reconstruction, while Poisson-shot preserves the high-frequency details better, and does not alter the identity of the ground truth person.



### C.5 SAMPLING SPEED 

As widely known in the literature, diffusion model-based methods are heavily dependent on the number of neural function evaluations (NFE). We investigate the performance in terms of LPIPS 

<div style="text-align: center;"><img src="imgs/img_in_chart_box_175_116_629_461.jpg" alt="Image" width="37%" /></div>


<div style="text-align: center;">(a) LPIPS vs. NFE </div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td>Method</td><td>Wall-clock time [s]</td></tr></thead><tbody><tr><td>Score-SDE (Song et al., 2021b)</td><td>36.71</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>2.029</td></tr><tr><td>MCG (Chung et al., 2022a)</td><td>80.10</td></tr><tr><td>PnP-ADMM (Chan et al., 2016)</td><td>3.631</td></tr><tr><td>BKS-styleGAN2 (Tran et al., 2021)</td><td>891.8</td></tr><tr><td>BKS-generic (Tran et al. 2021)</td><td>93.23</td></tr><tr><td>ER (Fienup, 1982)</td><td>5.604</td></tr><tr><td>HIO (Fienup & Dainty, 1987)</td><td>6.317</td></tr><tr><td>OSS (Rodriguez et all., 2013)</td><td>15.65</td></tr><tr><td>Ours</td><td>78.52</td></tr></tbody></table></body></html></div>


<div style="text-align: center;">(b) Runtime for each algorithm in Wall-clock time: Computed with a single GTX 2080Ti GPU. </div>


<div style="text-align: center;">Figure 11: Ablation studies performed with $\mathtt{S R}{\times}4$ task on FFHQ $256×256$ data, and the runtime analysis of the different algorithms. </div>


with respect to the change in NFEs in Fig.1la. For the experiment, we take the case of noisy $\mathtt{S R}{\times}4$ which is a problem where DDRM tends to perform well, in contrast to other problems, e.g. inpainting.In the high NFE regime $(\geq250)$ 1, DPS outperforms all the other methods, whereas in the low NFE regime $\widetilde{(\leq100)}$ , DDRM takes over. This can be attributed to DDIM (Song et al., 2021a) sampling strategy that DDRM adopts, known for better performance in the low NFE regimes. Orthogonal to the direction presented in this work, devising a method to improve the performance of DPS in such regime with advanced samplers (e.g. Lu et al. (2022); Liu et al. (2022)) would benefit the method.

### C.6 LIMITATIONS 

Inheriting the characteristics of the diffusion model-based methods, the proposed method is relatively slow, as can be seen in the runtime analysis of Fig. 11b. However, we note that our method is still faster than the GAN-based optimization methods, as we do not have to finetune the network itself.Moreover, the slow sampling speed could be mitigated with the incorporation of advanced samplers.Our method tends to preserve the high frequency details (e.g. beard, hair, texture) of the image, while methods such as DDRM tends to produce rather blurry images. In the qualitative view, and in the perception oriented metris (i.e. FID, LPIPS), our method clearly outperforms DDRM. In contrast, in standard distortion metrics such as PSNR, our method underperforms DDRM. This can be explained by the perception-distortion tradeoff phenomena (Blau & Michaeli, 2018), where preserving high frequency details may actually penalize the reconstructions from having better distortion metrics.Finally, we note that the reconstruction quality of phase retrieval is not as robust as compared to other problems – linear inverse problems and nonlinear deblurring. Due to the stochasticity, we often encounter failures among the posterior samples, which can be potentially counteracted by simply taking multiple samples, as was done in other methods. Devising methods to stabilize the samplers,especially for nonlinear phase retrieval problems, would be a promising direction of research.

## D EXPERIMENTAL DETAILS 

### D.1 IMPLEMENTATION DETAILS 

Step size. Here, we list the step sizes used inour DPS algorithm for each problem setting.

## • Linear inverse problem 

– Gaussian measurement noise 

*FFHQ 

.Super-resolution:$\zeta_{i}=1/\|\mathbf{\mathbf{\mathbf{\mathbf{\mathit{y}}}}}-\mathcal{A}(\hat{\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}}_{0}(\mathbf{\mathbf{\mathbf{\mathbf{\mathit{x}}}}}_{i}))\|$ 

. Inpainting:$\zeta_{i}=1/\|\pmb{y}-\mathcal{A}(\hat{\pmb{x}}_{0}(\pmb{x}_{i}))\|$ 

a : Deblurring (Gauss):$\zeta_{i}=1/\|\pmb{y}-\mathcal{A}(\hat{\pmb{x}}_{0}(\pmb{x}_{i}))\|$ 

a .  Deblurring (motion):$\zeta_{i}=1/\|\pmb{y}-\mathcal{A}(\hat{\pmb{x}}_{0}(\pmb{x}_{i}))\|$ 

$$\begin{aligned}{*}&{{}{~\operatorname{I m a g e N e t}~}}\\ {}&{{}\cdot{~\operatorname{S u p e r-r e s o l u o n:}~}\zeta_{i}=1/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{I n a p i l i n g:}~}\zeta_{i}=1/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{D e b l u r i n g}~}({\operatorname{G a u s s}}){:~}\zeta_{i}=0.4/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{D e b l u r i n g}~}({\operatorname{m o t i o n}}){:~}\zeta_{i}=0.6/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{P o i s o n~m e a s u r e m e n t~n o i s e}~}}\\ {}&{{}\cdot{~\operatorname{F F H Q}~}}\\ {}&{{}\cdot{~\operatorname{S u p e r-r e s o l u o n:}~}\zeta_{i}=0.3/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{D e b l u r i n g}~}({\operatorname{G a u s s}}){:~}\zeta_{i}=0.3/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{D e b l u r i n g}~}({\operatorname{m o t i o n}}){:~}\zeta_{i}=0.3/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{D e b l u r i n g~}~}({\operatorname{m o t i o n}}){:~}\zeta_{i}=0.3/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{O n o l i n e a r~i n v e r s e~p o l u e m e}~}}\\ {}&{{}\cdot{~\operatorname{G a u s s i n m~m e a s u r e m e n t~n o i s e}~}}\\ {}&{{}\cdot{~\operatorname{F F H Q}~}}\\ {}&{{}\cdot{~\operatorname{P h a s e~r e t i v e a l:}~}\zeta_{i}=0.4/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ {}&{{}\cdot{~\operatorname{n o n-u i f o r m~d e l u r i n g:}~}\zeta_{i}=1.0/\|\mathbf{y}-\mathcal{A}(\hat{x}_{0}(x_{i}))\|}\\ \end{aligned}$$

Score functions used. Pre-trained score function for the FFHQ dataset was taken from Choi et al.(2021)7, and the score function for the ImageNet dataset was taken from Dhariwal & Nichol (2021)8.Compute time. All experiments were performed on a single RTX 2080Ti GPU. FFHQ experiments take about 95 seconds per image (1000 NFE), while ImageNet experiments take about 600 seconds per image (1000 NFE) for reconstruction due to the much larger network size.

Code availability.Code is available at https://github.com/DPS2022/diffusion-posterior-sampling.



### D.2COMPARISON METHODS 

For DDRM, MCG, Score-SDE, and our method we use the same checkpoint for the score functions.

DDRM. All experiments were performed with the default setting of $\eta_{B}=1.0,\eta=0.85$ ,and leveraging DDIM (Song et al., 2021a) sampling for 20 NFEs. For the Gaussian deblurring experiment,the forward model was implemented by separable 1D convolutions for effi cient SVD.

MCG. We set the same values of α that are used in our methods (DPS). At each step, the additional data consistency steps are applied as Euclidean projections onto the measurement set 

$\mathcal{C}:=\{\mathbf{\mathit{x}}_{i}|\mathcal{A}(\mathbf{\mathit{x}}_{i})=\mathbf{\mathit{y}}_{i},\:\mathbf{\mathit{y}}_{i}\sim p(\mathbf{\mathit{y}}_{i}|\mathbf{\mathit{y}}_{0})\}$ 



Score-SDE. Score-SDE solves inverse problems by iteratively applying denoising followed by data consistency projections. As in MCG, we apply Euclidean projections onto the measurment set C.

PnP-ADMM. We take the implementation from the scico library (Balke et al., 2022). The parameters are set as follows:$\rho=0.2(\mathrm{ADMM})$ penalty parameter), ma $\mathtt{x i t e r}{=}12$ . For proximal mappings, we utilize the pretrained DnCNN Zhang et al. (2017) denoiser.

ADMM-TV. We minimize the following objective 

$$\operatorname*{m i n}_{{\mathbf{\mathit{x}}}}\frac{1}{2}\|{\mathbf{\mathit{y}}}-{\mathcal{A}}({\mathbf{\mathit{x}}}_{0})\|_{2}^{2}+\lambda\|{\mathbf{\mathit{D}}}{\mathbf{\mathit{x}}}_{0}\|_{2,1},$$

where $\mathbf{\overline{{\mathbf{\mathit{D}}}}}=[\mathbf{\overline{{\mathbf{\mathit{D}}}}}_{x},\mathbf{\overline{{\mathbf{\mathit{D}}}}}_{y}]$ computes the finite difference with respect to both axes, λ is the regularization weight, and $\|\cdot\|_{2,1}$ implements the isotropic TV regularization. Note that the optimization is solved with ADMM, and hence we have an additional parameter $\rho_{i}$ We take the implementation from the ${\tt S C\dot{I}C C}$ library (Balke et al., 2022). The parameters $\lambda,\rho$ were found with grid search for each optimization problems. We use the following settings:$(\lambda,\widetilde{\rho})=(2.7e-2,1.4\widetilde{e-1})$ for deblurring,$(\hat{\lambda},\rho)=(2.7\overset{\centerdot}{e}-2,1.0e-2)$ for SR and inpainting.



ER, HIO, OSs. For all algorithms, we initialize a real signal by sampling from the normal distribution as the problem statement of (Fienup, 1982). For the object domain constraint, we apply both the 

non-negative constraint and the finite support constraint. We set the number of iterations to 10,000for sufficient convergence. To mitigate the instability of reconstruction depending on initialization,we repeat each algorithm four times per data and report the best one with the smallest mean squared error between the measurement and amplitude of the estimation in the Fourier domain. In the case of HIO and OSS, we set β to 0.9, which yields best results.



## EFURTHER EXPERIMENTAL RESULTS 

<div style="text-align: center;">We first provide quantitative evaluations based on the standard PSNR and SSIM metrics in Table 6and Table 7. </div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td></td><td colspan="2">SR (×4)</td><td colspan="2">Inpaint (box)</td><td colspan="2">Inpaint (random)</td><td colspan="2">Deblur (gauss)</td><td colspan="2">Deblur (motion)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM↑</td></tr></thead><tbody><tr><td>DPS (ours)</td><td>25.67</td><td>0.852</td><td>22.47</td><td>0.873</td><td>25.23</td><td>0.851</td><td>24.25</td><td>0.811</td><td>24.92</td><td>0.859</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>25.36</td><td>0.835</td><td>22.24</td><td>0.869</td><td>9.19</td><td>0.319</td><td>23.36</td><td>0.767</td><td>-</td><td>=</td></tr><tr><td>MCG (Chung et al., 2022a)</td><td>20.05</td><td>0.559</td><td>19.97</td><td>0.703</td><td>21.57</td><td>0.751</td><td>6.72</td><td>0.051</td><td>6.72</td><td>0.055</td></tr><tr><td>PnP-ADMM (Chan et al., 2016)</td><td>26.55</td><td>0.865</td><td>11.65</td><td>0.642</td><td>8.41</td><td>0.325</td><td>24.93</td><td>0.812</td><td>24.65</td><td>0.825</td></tr><tr><td>Score-SDE (Song et al., 2021b) (ILVR (Choi et al., 2021))</td><td>17.62</td><td>0.617</td><td>18.51</td><td>0.678</td><td>13.52</td><td>0.437</td><td>7.12</td><td>0.109</td><td>6.58</td><td>0.102</td></tr><tr><td>ADMM-TV</td><td>23.86</td><td>0.803</td><td>17.81</td><td>0.814</td><td>22.03</td><td>0.784</td><td>22.37</td><td>0.801</td><td>21.36</td><td>0.758</td></tr></tbody></table></body></html></div>


<div style="text-align: center;">Table6: QuaevuioSR,Io olvigiaverobms FHQ $256{\times}256{\cdot}$ 1k validation dataset. Bold: best, underline: second best. </div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td></td><td colspan="2">SR $(\times4)$</td><td colspan="2">Inpaint (box)</td><td colspan="2">Inpaint (random)</td><td colspan="2">Deblur (gauss)</td><td colspan="2">Deblur (motion)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr></thead><tbody><tr><td>DPS (ours)</td><td>23.87</td><td>0.781</td><td>18.90</td><td>0.794</td><td>22.20</td><td>0.739</td><td>21.97</td><td>0.706</td><td>20.55</td><td>0.634</td></tr><tr><td>DDRM (Kawar et al., 2022)</td><td>24.96</td><td>0.790</td><td>18.66</td><td>0.814</td><td>14.29</td><td>0.403</td><td>22.73</td><td>0.705</td><td>-</td><td>-</td></tr><tr><td>MCG (Chung et al., 2022a)</td><td>13.39</td><td>0.227</td><td>17.36</td><td>0.633</td><td>19.03</td><td>0.546</td><td>16.32</td><td>0.441</td><td>5.89</td><td>0.037</td></tr><tr><td>PnP-ADMM (Chan et al., 2016)</td><td>23.75</td><td>0.761</td><td>12.70</td><td>0.657</td><td>8.39</td><td>0.300</td><td>21.81</td><td>0.669</td><td>21.98</td><td>0.702</td></tr><tr><td>Score-SDE (Song et al., 2021b) (ILVR (Choi et al., 2021))</td><td>12.25</td><td>0.256</td><td>16.48</td><td>0.612</td><td>18.62</td><td>0.517</td><td>15.97</td><td>0.436</td><td>7.21</td><td>0.120</td></tr><tr><td>ADMM-TV</td><td>22.17</td><td>0.679</td><td>17.96</td><td>0.785</td><td>20.96</td><td>0.676</td><td>19.99</td><td>0.634</td><td>20.79</td><td>0.677</td></tr></tbody></table></body></html></div>


<div style="text-align: center;">Table 7: Quantitative evaluation (PSNR, SSIM) of solving linear inverse problems on ImageNet $256\times256-1k$ validation dataset. Bold: best, underline: second best. </div>


Further experimental results that show the ability of our method to sample multiple reconstructions are presented in Figs. 12,13, 14, 15, 16, 17 (Gaussian measurement with $\sigma=0.05)$ ), and Fig. 18,19(Poisson measurement with $\lambda=1.0)$ ).



<div style="text-align: center;"><img src="imgs/img_in_image_box_130_317_1162_1365.jpg" alt="Image" width="84%" /></div>


<div style="text-align: center;">Figure 12: SR $(\mathbf{L e f t}\times8$ , Right ×16), results on the FFHQ (Karras et al., 2019)$256\times256$ dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_127_296_1104_1363.jpg" alt="Image" width="79%" /></div>


<div style="text-align: center;">Figure 13: SR $(\mathbf{L e f t}\times8$ $\mathrm{R i g h t}\times16)$ , results on the ImageNet (Deng et al, 2009)$256\times256$ dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_127_195_1132_1424.jpg" alt="Image" width="82%" /></div>


<div style="text-align: center;">Figure 14: Inpainting results (Left $128\times128$ box, Right 92% random) on the FFHQ (Karras et al.,2019)$256\times256$ dataset. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_137_197_1137_1424.jpg" alt="Image" width="81%" /></div>


<div style="text-align: center;">Figure 15: Inpainting results (Left $128\times128$ box, Right 92% random) on the ImageNet (Deng et all.,2009)$256\times256$ dataset. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_129_284_1157_1347.jpg" alt="Image" width="83%" /></div>


<div style="text-align: center;">Figure 16: Deblurring results (Left Gaussian, Right motion) on the FFHQ (Karras et al, 2019)$256\times256$ dataset. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_132_286_1131_1351.jpg" alt="Image" width="81%" /></div>


<div style="text-align: center;">Figure7:tGINl.,$256\times256$ dataset. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_128_294_1128_1345.jpg" alt="Image" width="81%" /></div>


<div style="text-align: center;">Figure 18: SR $(\mathbf{L e f t}\times8,\mathbf{R i g h t}\times16)$ with Poisson noise $(\lambda=0.05)$ 1, results on the FFHQ (Karras et al., 2019) 256 × 256 dataset. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_131_287_1140_1352.jpg" alt="Image" width="82%" /></div>


<div style="text-align: center;">Figure 19: Deblurring results with Poisson noise $(\lambda=1.0)$ (Left Gaussian, Right motion) on the FFHQ (Karras et al., 2019)$256\times256$ dataset. </div>
