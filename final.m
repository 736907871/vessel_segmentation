%% TEST Frangi_filter  
clc;close all;clear all;  
%numpΪͼƬ�ļ�����������ֵ��21����Сֵ����������������������·��Ϊ�½��ļ���H:\learn and work\ͼ����\��Ŀ��ҵ\DIP����Ĥ�ָ�գ�\DRIVE\DRIVE\training\result\
nump=21;

for i=21:nump
J=imread(strcat('H:\learn and work\ͼ����\��Ŀ��ҵ\DIP����Ĥ�ָ�գ�\DRIVE\DRIVE\training\mask\',num2str(i),'_training_mask.gif'));
I=imread(strcat('H:\learn and work\ͼ����\��Ŀ��ҵ\DIP����Ĥ�ָ�գ�\DRIVE\DRIVE\training\images\',num2str(i),'_training.tif'));

img=I;

%ѡ����ɫͨ��
Ir = I(:,:,1);%��ͨ��
Ig = I(:,:,2);%��ͨ��
Ib = I(:,:,3);%��ͨ��

%figure,subplot(121),imshow(double(rgb2gray(I)),[]);title('ԭ�Ҷ�ͼ'); 
%����Ӧֱ��ͼ����
I=double(adapthisteq(Ig)); 
%subplot(122),imshow(I,[]);title('����Ӧֱ��ͼ����'); 

Ivessel=(FrangiFilter2D(I));
%figure,subplot(121), imshow(Ivessel,[]);title('Frangi�˲����')  

%ȥ�� �ϵ�
Ivessel = medfilt2(Ivessel,[3 3]);

%SE=strel('rectangle',[3 3]);      
%Ivessel=imclose(Ivessel,SE); 
%figure,imshow(Ivessel);title('�ղ���');


%��ֵ��
Ivessel= 1 * (Ivessel .^ 0.32); 
threshold2 = graythresh(Ivessel);
binary_data1 = im2bw(Ivessel,threshold2);%��ͼ����ж�ֵ��
%figure,subplot(131), imshow(binary_data1);
%title('Ѫ�ܶ�ֵͼ');

%��ֵ�˲�ȥ������
for i2=0:2
    binary_data1=medfilt2(binary_data1,[3 3]);
end
%figure,imshow(binary_data1,[]);

%% ȥ��������
img=double(rgb2gray(img));
[m n] = size(img);
img_hist = zeros(1,256);
for i3 = 1:m
    for j = 1:n
        img_hist(img(i3, j)+1) = img_hist(img(i3, j)+1) + 1;
    end
end
img_hist_pro = img_hist/m/n;         %�Ҷȼ������ܶȷֲ�
sigma2_max = 0;threshold = 0;
for t = 0:255
    w0 = 0;w1 = 0; u0 = 0; u1 = 0; u = 0;
    for q = 0:255
        if q <= t
            w0 = w0 + img_hist_pro(q+1);
            u0 = u0 + (q)*img_hist_pro(q+1);
        else
            w1 = w1 + img_hist_pro(q+1);
            u1 = u1 + (q)*img_hist_pro(q+1);
        end
    end
    u = u0 + u1;
    u0 = u0 / (w0+eps);
    u1 = u1 / (w1+eps);
    sigma2 = w0 * (u0 - u)^2 + w1 * (u1 - u)^2;     %��ȡ�����С
    if (sigma2 > sigma2_max)
        sigma2_max = sigma2;
        threshold = t;
    end
end
img_out = img;
for i4 = 1:m                                         %��ֵ��
    for j = 1:n
        if img(i4, j) >= threshold
            img_out(i4, j) = 255;
        else 
            img_out(i4, j) = 0;
        end
    end
end


%subplot(132);
%imshow(img_out);
%title('�����Ĥ');

new_img = imsubtract(Ivessel,double(img_out));%�����Ҷ�ͼ

threshold = graythresh(new_img);
SE=strel('disk',6);      
new_img=imdilate(new_img,SE); %��ֵ�˲�֮��,����
%subplot(133);
binary_data2 = im2bw(new_img,threshold);%��ͼ����ж�ֵ��,�ֶ�����
%imshow(binary_data2);
%title('������ֵͼ');

binary_data = binary_data1-binary_data2;%
binary_data(binary_data==-1)=0;

figure,imshow(binary_data);
title('���յĶ�ֵͼ');
%imwrite(binary_data,['H:\learn and work\ͼ����\��Ŀ��ҵ\DIP����Ĥ�ָ�գ�\DRIVE\DRIVE\training\result\',num2str(i),'.tif'])

end





%frangi
%���Կ���A�Թ��������������ã�B�Ա������������ã����ʣ�µ�ֻ��Ѫ�ܴ����ź���Ӧǿ��
%����˲���ֻ���ھ���߶Ⱥ�Ѫ�ܿ����ӽ���ʱ��Ч����á����ȷ������ߴ��أ���ֱ��Ҳ������Ч�ķ�������--ö�ٷ���
%���Գ����о����ò�ͬ�ľ���߶�ȥ���˲����õ��Ķ���˲���ͼ���У���ÿһ�㴦ѡ����Ӧֵ��ߵĽ���������С�FrangiScaleRange������ö�ٵĳ߶ȷ�Χ��
function [outIm,whatScale,Direction] = FrangiFilter2D(I, options)  
defaultoptions = struct('FrangiScaleRange', [0.05 5], 'FrangiScaleRatio', 0.05 , 'FrangiBetaOne', 0.5, 'FrangiBetaTwo', 15, 'verbose',true,'BlackWhite',true);  

% Process inputs  
if(~exist('options','var')) 
    options=defaultoptions;   
else  
    tags = fieldnames(defaultoptions);  
    for i=1:length(tags)  
         if(~isfield(options,tags{i})),  options.(tags{i})=defaultoptions.(tags{i}); end  
    end  
    if(length(tags)~=length(fieldnames(options)))
        warning('FrangiFilter2D:unknownoption','unknown options found');  
    end  
end  
  
%sigam��Χ��FrangiScaleRange��1����FrangiScaleRange��2��������ΪFrangiScaleRatio������Ҫ���ͼ��  
  
sigmas=options.FrangiScaleRange(1):options.FrangiScaleRatio:options.FrangiScaleRange(2);  
sigmas = sort(sigmas, 'ascend');  
  
beta  = 2*options.FrangiBetaOne^2;  
c     = 2*options.FrangiBetaTwo^2;  
  
% Make matrices to store all filterd images  
ALLfiltered=zeros([size(I) length(sigmas)]);  
ALLangles=zeros([size(I) length(sigmas)]);  
  
% Frangi filter for all sigmas  
for i = 1:length(sigmas)
    % Show progress  
    if(options.verbose)  
        disp(['Current Frangi Filter Sigma: ' num2str(sigmas(i)) ]);  
    end  
      
    % Make 2D hessian  
    [Dxx,Dxy,Dyy] = Hessian2D(I,sigmas(i));  
      
    % Correct for scale  
    Dxx = (sigmas(i)^2)*Dxx;  
    Dxy = (sigmas(i)^2)*Dxy;  
    Dyy = (sigmas(i)^2)*Dyy;  
     
    % Calculate (abs sorted) eigenvalues and vectors  
    [Lambda2,Lambda1,Ix,Iy]=eig2image(Dxx,Dxy,Dyy);  
  
    % Compute the direction of the minor eigenvector  
    angles = atan2(Ix,Iy);  
  
    % Compute some similarity measures  
    Lambda1(Lambda1==0) = eps;  
    Rb = (Lambda2./Lambda1).^2;  
    S2 = Lambda1.^2 + Lambda2.^2;  
     
    % Compute the output image  
    Ifiltered = exp(-Rb/beta) .*(ones(size(I))-exp(-S2/c));  
      
    % see pp. 45  
    if(options.BlackWhite)  
        Ifiltered(Lambda1<0)=0;  
    else  
        Ifiltered(Lambda1>0)=0;  
    end  
    % store the results in 3D matrices  
    ALLfiltered(:,:,i) = Ifiltered;  
    ALLangles(:,:,i) = angles;  
end  
  
% Return for every pixel the value of the scale(sigma) with the maximum   
% output pixel value  
if length(sigmas) > 1
    [outIm,whatScale] = max(ALLfiltered,[],3);  
    outIm = reshape(outIm,size(I));  
    if(nargout>1)  
        whatScale = reshape(whatScale,size(I));  
    end  
    if(nargout>2)  
        Direction = reshape(ALLangles((1:numel(I))'+(whatScale(:)-1)*numel(I)),size(I));  
    end  
else  
    outIm = reshape(ALLfiltered,size(I));  
    if(nargout>1)  
            whatScale = ones(size(I));  
    end  
    if(nargout>2)  
        Direction = reshape(ALLangles,size(I));  
    end  
end  
end


%��Hessian���󣺶�Ӧ���� Hessian2D()
function [Dxx,Dxy,Dyy] = Hessian2D(I,Sigma)

if nargin < 2, Sigma = 1; end

[X,Y]   = ndgrid(-round(3*Sigma):round(3*Sigma));

DGaussxx = 1/(2*pi*Sigma^4) * (X.^2/Sigma^2 - 1) .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussxy = 1/(2*pi*Sigma^6) * (X .* Y)           .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussyy = DGaussxx';

Dxx = imfilter(I,DGaussxx,'conv');
Dxy = imfilter(I,DGaussxy,'conv');
Dyy = imfilter(I,DGaussyy,'conv');
end

%��Hessian�������������ֵ����Ӧ���� eig2image()
function [Lambda1,Lambda2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)

tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
v2x = 2*Dxy; v2y = Dyy - Dxx + tmp;

mag = sqrt(v2x.^2 + v2y.^2); i = (mag ~= 0);
v2x(i) = v2x(i)./mag(i);
v2y(i) = v2y(i)./mag(i);

v1x = -v2y; 
v1y = v2x;

mu1 = 0.5*(Dxx + Dyy + tmp);
mu2 = 0.5*(Dxx + Dyy - tmp);

check=abs(mu1)>abs(mu2);

Lambda1=mu1; Lambda1(check)=mu2(check);
Lambda2=mu2; Lambda2(check)=mu1(check);

Ix=v1x; Ix(check)=v2x(check);
Iy=v1y; Iy(check)=v2y(check);
end