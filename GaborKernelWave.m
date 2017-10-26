function [Gr, Gi] = GaborKernelWave(GaborH, GaborW, U, V, Kmax, f, sigma, orientation, flag)
% function [Gr, Gi] = GaborKernelWave[GaborH, GaborW, U, V, Kmax, f, sigma, orientation, flag]
% 
% The Gabor wavelets (kernels, filters) can be defined as follows: 
%            ||Ku,v||^2
% G(Z) = ---------------- exp((-||Ku,v||^2 * Z^2)/(2*sigma*sigma))*(exp(i*Ku,v*Z)-exp(-sigma*sigma/2))
%          sigma*sigma
%
% || || denotes the norm operator
% Z = (x,y)
% Gr: The real part of the Gabor kernels
% Gi: The imaginary part of the Gabor kernels
% GaborH, GaborW: the width and length of the Gabor window
% U define the orientation of the Gabor kernels, for instance, U = {0,1,...,7}
% V define the scale of the Gabor kernels, for instance, v = {0,1,...,4}
% Kmax is the maximum frequency
% f is the spacing factor between kernels in the frequency domain
% sigma defines the Gaussian envelope along x and y axes
% orientation is the total number of orientations
% flag : 1 -> remove the dc value of Gabor
%        0 -> not to remove
% 
HarfH = fix(GaborH/2);
HarfW = fix(GaborW/2);

Qu = pi*U/orientation;
sqsigma = sigma*sigma;

Kv = Kmax/(f^V);

if (flag == 1)
    postmean = exp(-sqsigma/2);
else
    postmean = 0;
end
% Gr = zeros(GaborH,GaborW);
% Gi = zeros(GaborH,GagorW);
for x = -HarfH : HarfH
    for y =  -HarfW : HarfW
        tmp1 = exp(-(Kv*Kv*(x*x+y*y)/(2*sqsigma)));
        tmp2 = cos(Kv*cos(Qu)*x+Kv*sin(Qu)*y) - postmean;        
        tmp3 = sin(Kv*cos(Qu)*x+Kv*sin(Qu)*y);
        
        Gr(x + HarfH + 1, y + HarfW + 1) = Kv*Kv*tmp1*tmp2/sqsigma;
        Gi(x + HarfH + 1, y + HarfW + 1) = Kv*Kv*tmp1*tmp3/sqsigma;
    end
end