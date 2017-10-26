function [f V] = PCANet_train(InImg,PCANet,IdtExt)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)  
% PCANet    PCANet parameters (struct)
%       .PCANet.NumStages      
%           the number of stages in PCANet; e.g., 2  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., 3, 5, 7
%           only a odd number allowed
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize 
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio 
%           overlapped block region ratio; e.g., 0 means no overlapped 
%           between blocks, and 0.3 means 30% of blocksize is overlapped 
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise  
%
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% V         learned PCA filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% ========= CITATION ============
%
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" 
% submitted to IEEE TPAMI. 
% ArXiv eprint: http://arxiv.org/abs/1404.3606 

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!

if length(PCANet.NumFilters)~= PCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InImg);

V = cell(PCANet.NumStages,1); 
OutImg = InImg; 
ImgIdx = (1:NumImg)';

for stage = 1:PCANet.NumStages
    display(['Computing PCA filter bank and its outputs at stage ' num2str(stage) '...'])
    % compute PCA filter banks
    V{stage} = PCA_FilterBank(OutImg, PCANet.PatchSize, PCANet.NumFilters(stage)); 
    
    % compute the PCA outputs only when it is NOT the last stage
    if stage ~= PCANet.NumStages 
        [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
            PCANet.PatchSize, PCANet.NumFilters(stage), V{stage});  
    end
end

 
f = cell(NumImg,1); % compute the PCANet training feature one by one 
parfor idx = 1:NumImg
    if 0==mod(idx,100); 
        display(['Extracting PCANet feature of the ' num2str(idx) ...
            'th training sample...']); 
    end

    % select feature maps corresponding to image "idx" 
    % (outputs of the-last-but-one PCA filter bank) 
    OutImgIndex = ImgIdx==idx; 

    % compute the last PCA outputs of image "idx"
    [OutImg_i ImgIdx_i] = PCA_output(OutImg(OutImgIndex), ...
        ones(sum(OutImgIndex),1),...
        PCANet.PatchSize, PCANet.NumFilters(end), V{end}); 

    % compute the feature of image "idx"
    [f{idx} BlkIdx] = HashingHist(PCANet,ImgIdx_i,OutImg_i); 
    % OutImg(OutImgIndex) = cell(sum(OutImgIndex),1); 

end
    
f = [f{:}];








