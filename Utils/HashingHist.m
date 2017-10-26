function [f BlkIdx] = HashingHist(PCANet,ImgIdx,OutImg)
% Output layer of PCANet (Hashing plus local histogram)
% ========= INPUT ============
% PCANet  PCANet parameters (struct)
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
% ImgIdx  Image index for OutImg (column vector)
% OutImg  PCA filter output before the last stage (cell structure)
% ========= OUTPUT ===========
% f       PCANet features (each column corresponds to feature of each image)
% BlkIdx  index of local block from which the histogram is compuated
% ========= CITATION ============
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" submitted to IEEE TPAMI. 
% ArXiv eprint: http://arxiv.org/abs/1404.3606 

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!

addpath('./Utils')

NumImg = max(ImgIdx);
f = cell(NumImg,1);
map_weights = 2.^((PCANet.NumFilters(end)-1):-1:0); % weights for binary to decimal conversion
for Idx = 1:NumImg
  
    Idx_span = find(ImgIdx == Idx);
    Bhist = cell(length(Idx_span),1);
    NumImginO = length(Idx_span)/PCANet.NumFilters(end); % the number of feature maps in "\cal O"
    
    for i = 1:NumImginO 
        
        T = zeros(size(OutImg(Idx_span(1))));
        
        for j = 1:PCANet.NumFilters(end)
            T = T + map_weights(j)*Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)}); 
            % weighted combination; hashing codes to decimal number conversion
            
            OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)} = [];
        end
        
        Bhist{i} = sparse(histc(im2col_general(T,PCANet.HistBlockSize,...
            round((1-PCANet.BlkOverLapRatio)*PCANet.HistBlockSize)),(0:2^PCANet.NumFilters(end)-1)')); 
        % calculate histogram for each local block in "T"
        
        Bhist{i} = bsxfun(@times, Bhist{i}, ...
            2^PCANet.NumFilters(end)./sum(Bhist{i})); % to ensure that sum of each block-wise histogram is equal 
    end           
    f{Idx} = vec([Bhist{:}]);
end
f = [f{:}];
BlkIdx = kron(ones(NumImginO,1),kron((1:size(Bhist{1},2))',ones(size(Bhist{1},1),1)));

%-------------------------------
function X = Heaviside(X) % binary quantization
X = sign(X);
X(X<=0) = 0;

function x = vec(X) % vectorization
x = X(:);