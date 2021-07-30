%% TEST 
clc;close all;clear all;  
%nump为图片文件名中序号最大值，21是最小值，用于批量操作，保存结果路径为新建文件夹C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\result\
nump=21;

accuracy=0;
spec=0;
sens=0;
acc=0;

for t=21:nump
J=imread(strcat('C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\result\',num2str(t),'.tif'));
I=imread(strcat('C:\Users\Admin\Desktop\DIP\DRIVE\DRIVE\training\1st_manual\',num2str(t),'_manual1.gif'));
%length=numel(I(:));

%计算结果与专家的差值  黑色部分（0）代表是相同部分，白色部分为不同部分
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
% 查准率(Precision)：所有诊断为患病（1）样本中实际为患病的比率。
% 召回率(Recall)：所有患病样本中被发现并诊断为患病的比率。

%准确率（accuracy）： (TP + TN )/( TP + FP + TN + FN)
accuracy=accuracy+(TP + TN )/( TP + FP + TN + FN);
% 查准率\精准率（precision）：TP / (TP + FP)，正确预测为正占全部预测为正的比例
acc=acc+TP/(TP+FP);
% 召回率\敏感性 = TP/P = TP/(TP+FN),正确预测为正占全部正样本的比例
sens=sens+TP/(TP+FN);
% 特异性 = TN/N = TN/(TN+FP),样本中的负类被预测为负类的比例(正确预测为负占全部负样本的比例)
spec=spec+TN/(TN+FP);

end
accuracy_average=accuracy/(nump-20);
acc_average=acc/(nump-20);
sense_average=sens/(nump-20);
spec_average=spec/(nump-20);





