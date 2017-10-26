% The files are the MATLAB source code for the paper:
% Feng Gao,  Junyu Dong, Bo Li, Qizhi Xu. 
% Automatic Change Detection in Synthetic Aperture Radar Images Based on PCANet 
% IEEE Geoscience and Remote Sensing Letters, 2016, 13(12), 1792-1796.
%
% The demo has not been well organized. Please contact me if you meet any problems.
% 
% Email: gaofeng@ouc.edu.cn
% 
% 注意事项：
%  1. 因为训练时样本是随机生成，样本不同会影响最终的结果。
%     所以每次运行的结果都不会一样。
%     作者会在后期发布的版本中进行修正（固定生成随机数的种子）
%  2. 代码运行速度较慢，建设开启多核并行
%     Matlab 2012 中为 matlabpool open
%     在Intel Xeon处理器上处理速度还可以

clear;
clc;
close all;

addpath('./utils');
addpath('./liblinear');

im1   = imread('./pic/san_1.bmp');
im2   = imread('./pic/san_2.bmp');
im_gt = imread('./pic/san_gt.bmp');

im1   = double(im1(:,:,1));  
im2   = double(im2(:,:,1)); 
im_gt = double(im_gt(:,:,1));


% 求差分图像，得到差分图像后进行了滤波预处理 
fprintf('... ... compute the difference image ... ...\n');
im_di = di_gen(im1, im2);


% setting of variables
% 这些参数来自 GaborTLC 的一些默认参数
HW = 21;
GaborH = HW;
GaborW = HW;

sigma = 2.8*pi;

Kmax = 2.0*pi;
f = sqrt(2);
flag = 1;
scale = 5;
orientation = 8;
V = 0:1:(scale-1);
U = 0:1:(orientation-1);

[Ylen,Xlen] = size(im_di);
% the total number of pixels for the difference image
pixel_sum = Ylen*Xlen;
% the total number of self-similar Gabor filters
subgraph_sum = scale*orientation;

% Initialization
GaImout_MAGNITUDE = cell(scale,orientation);          % magnitude

% Step 2): Feature extraction==============================================
% Gabor wavelet transform
for s = 1:scale,
    for n = 1:orientation,
        GaImout_MAGNITUDE{s,n} = zeros(Ylen,Xlen);
        [Gr,Gi] = GaborKernelWave(GaborH, GaborW, U(n), V(s), Kmax, f, sigma, orientation, flag);
        % Gr: The real part of the Gabor kernels
        % Gi: The imaginary part of the Gabor kernels
        Regabout = conv2(im_di,double(Gr),'same');
        Imgabout = conv2(im_di,double(Gi),'same');
        % Magnitude
        GaImout_MAGNITUDE{s,n} = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);    
    end;
end;

clear GaborH GaborW sigma;
clear Gi Gr HW Kmax;
clear Imgabout Regabout;
clear U V f flag;

% acquire the feature vector of each pixel for the difference image
% 1): all amplitudes at different scales and orientations
pixel_vector_1 = zeros(pixel_sum,subgraph_sum);
k = 1;
for s = 1:scale
    for n = 1:orientation
        temp_gaimout_1 = zeros(Ylen,Xlen);
        temp_gaimout_1 = GaImout_MAGNITUDE{s,n};
        pixel_vector_1(:,k) = reshape(temp_gaimout_1',pixel_sum,1);
        k = k + 1;
    end
end

clear k n s GaImout_MAGNITUDE subgraph_sum;

% 2): maximum amplitudes for all orientations at different scales
pixel_vectorr_2 = zeros(pixel_sum,scale);
for s = 1:scale
    temp_gaimout_2 = zeros(pixel_sum,orientation);
    temp_gaimout_2 = pixel_vector_1(:,(s-1)*orientation+1:s*orientation);
    pixel_vector_2(:,s) = max(temp_gaimout_2,[],2);
end
clear scale temp_gaimout_1 temp_gaimout_2 pixel_vector_1;
clear s pixel_sum orientation;


% Step 3): Hierarchical clustering==========================
fprintf('... ... hclustering begin ... ...\n');
im_lab = HClustering(pixel_vector_2, im_di);
fprintf('@@@ @@@ hclustering finished @@@@\n');
% 初始化结果保存在 im_lab 中
% 变化的部分标记为 0
% 未变化的部分标记为 1
clear CM0 pixel_vector_2 im_di;


% PatSize 必须为奇数
PatSize = 5;

% PCANet 部分
gao_trn;
gao_tst;

% 把较小的噪声区域去掉
PreRes = reshape(PreRes, Ylen, Xlen);
[lab_pre,num] = bwlabel(~PreRes);

for i = 1:num
    idx = find(lab_pre==i);
    if numel(idx) <= 20
        lab_pre(idx)=0;
    end
end

lab_pre = lab_pre>0;
res = uint8(lab_pre)*255;
[FA,MA,OE,CA] = DAcom(im_gt, res);
% 保存结果
fid = fopen('rec.txt', 'a');
fprintf(fid, 'PatSize = %d\n', PatSize);
fprintf(fid, 'FA  : %d \n', FA);
fprintf(fid, 'MD  : %d \n', MA);
fprintf(fid, 'OE  : %d \n', OE);
fprintf(fid, 'PCC : %f \n\n\n', CA);
fclose(fid);

imwrite(res, 'chage_map.png');






















