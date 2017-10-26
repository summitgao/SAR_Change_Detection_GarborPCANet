


% ²âÊÔ½×¶Î
fprintf('\n ====== PCANet Testing ======= \n')
TstNum = numel(im);
PreRes = zeros(TstNum,1); 

parfor idx = 1:TstNum
    
    if mod(idx,200) == 0
        fprintf('   ... idx NO. %d processing ...\n', idx);
    end
    
    if im_lab(idx) == 1
        PreRes(idx) = 1;
        continue;
    elseif im_lab(idx) == 0
        PreRes(idx) = 0;
        continue;
    end
    
    ftest = PCANet_FeaExt(im(idx),V,PCANet);
    % label predictoin by libsvm
    [xLabel_est, accuracy, decision_values] = predict(0,...
        sparse(ftest'), models, '-q');
    if xLabel_est == 1
        PreRes(idx) = 1;
    end
    
end



    