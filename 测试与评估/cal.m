%% TEST 
clc;close all;clear all;  
%numpΪͼƬ�ļ�����������ֵ��21����Сֵ����������������������·��Ϊ�½��ļ���C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\result\
nump=21;

accuracy=0;
spec=0;
sens=0;
acc=0;

for t=21:nump
J=imread(strcat('C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\result\',num2str(t),'.tif'));
I=imread(strcat('C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\1st_manual\',num2str(t),'_manual1.gif'));
%length=numel(I(:));

%��������ר�ҵĲ�ֵ  ��ɫ���֣�0����������ͬ���֣���ɫ����Ϊ��ͬ����
sub=double(J)-double(I);

TP=0;
FP=0;
TN=0;
FN=0;

for i=1:size(I,1)
    for j=1:size(I,2)
        if(sub(i,j)==0)
            if(I(i,j)==255)
                TP=TP+1;
            else
                TN=TN+1;
            end
        end
        if(sub(i,j)==255)
                FP=FP+1;
        end
        if(sub(i,j)==-255)
            FN=FN+1;
        end
    end
end
% ��׼��(Precision)���������Ϊ������1��������ʵ��Ϊ�����ı��ʡ�
% �ٻ���(Recall)�����л��������б����ֲ����Ϊ�����ı��ʡ�

%׼ȷ�ʣ�accuracy���� (TP + TN )/( TP + FP + TN + FN)
accuracy=accuracy+(TP + TN )/( TP + FP + TN + FN);
% ��׼��\��׼�ʣ�precision����TP / (TP + FP)����ȷԤ��Ϊ��ռȫ��Ԥ��Ϊ���ı���
acc=acc+TP/(TP+FP);
% �ٻ���\������ = TP/P = TP/(TP+FN),��ȷԤ��Ϊ��ռȫ���������ı���
sens=sens+TP/(TP+FN);
% ������ = TN/N = TN/(TN+FP),�����еĸ��౻Ԥ��Ϊ����ı���(��ȷԤ��Ϊ��ռȫ���������ı���)
spec=spec+TN/(TN+FP);

end
accuracy_average=accuracy/(nump-20);
acc_average=acc/(nump-20);
sense_average=sens/(nump-20);
spec_average=spec/(nump-20);





