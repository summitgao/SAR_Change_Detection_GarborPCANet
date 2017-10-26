
addpath('./Utils');
addpath('./Liblinear');



%% PCANet parameters
PCANet.NumStages = 2;
PCANet.PatchSize = PatSize;
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [PatSize*2, PatSize]; 
PCANet.BlkOverLapRatio = 0;
PCANet.Pyramid = [];


fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

% 获取 lab 信息
pos_lab = find(im_lab == 1);
neg_lab = find(im_lab == 0);
% 对正负样本打乱顺序
pos_lab = pos_lab(randperm(numel(pos_lab)));
neg_lab = neg_lab(randperm(numel(neg_lab)));

[ylen, xlen] = size(im1);

% 图像周围填零，然后每个像素周围取Patch，保存
mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im1; 
im1 = im2col_general(imTmp, [PatSize, PatSize]);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im2; 
im2 = im2col_general(imTmp, [PatSize, PatSize]);
clear imTmp mag;
clear xlen ylen;

% 合并样本到 im
im1 = mat2imgcell(im1, PatSize, PatSize, 'gray');
im2 = mat2imgcell(im2, PatSize, PatSize, 'gray');
parfor idx = 1 : numel(im1)
    im{idx} = [im1{idx}; im2{idx}];    
end
clear im1 im2 idx;

% 随机选择全图像素百分之30的样本，按照比例筛选正负样本
[ylen, xlen] = size(im);
NumSam = round(ylen*xlen*0.30);
PosNum = round(NumSam*numel(pos_lab)/(numel(neg_lab) + numel(pos_lab)));
NegNum = NumSam - PosNum;
if NegNum > numel(neg_lab)
    NegNum = numel(neg_lab);
end


% 取出正负样本图像块
PosPat = im(pos_lab(1:PosNum));
NegPat = im(neg_lab(1:NegNum));
TrnPat = [PosPat, NegPat];
TrnLab = [ones(PosNum, 1); zeros(NegNum, 1)];


fprintf('  SamNum : %d\n', PosNum+NegNum);
fprintf('  PosNum : %d\n', PosNum);
fprintf('  NegNum : %d\n', NegNum);


[ftrain V] = PCANet_train(TrnPat,PCANet,1); 
clear NumSam PosNum NegNum BlkIdx;
clear PosPat NegPat TrnPat;
clear pos_lab neg_lab;


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% we use linear SVM classifier (C = 1), calling libsvm library
models = train(TrnLab, ftrain', '-s 1 -q'); 
clear ftrain TrnLab;



    