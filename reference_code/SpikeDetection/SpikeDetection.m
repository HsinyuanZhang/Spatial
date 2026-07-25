clear all
clc
%% Load Files 
[file,path]                 = uigetfile('../Database/*.h5','Select Wired-OR File');
recordingFile               = strcat(path, file);
outputPath                  = path;

%% Parameters

Fs                          = 20000;
ColNum                      = 32;
RowNum                      = 32;
nonCollisionCntrH           = 7;
preDetectionThr             = 3;
ampThreshold                = 0;
NEOThreshold                = 40;
centralChannelDetection     = 1;

startTime                   = 1;
batchSize                   = 1 * Fs;
sampleNum                   = 30 * Fs;

batchExtendLen              = 10;
dataIn                      = zeros(RowNum, ColNum, batchSize + batchExtendLen);

nonCollisionCntr            = uint8(zeros(RowNum, ColNum));
preDetectedSpike            = false(RowNum, ColNum, sampleNum);
detectedSpike               = false(RowNum, ColNum, sampleNum);


for b = 1 : sampleNum / batchSize
    disp(strcat(num2str(100 * b/(sampleNum / batchSize)), '%'));
    %% Spike Detection Process
    wiredOR                                   = permute(h5read(recordingFile,'/wiredOR', [(b - 1) * batchSize + 1, 1, 1], [batchSize, ColNum, RowNum]), [3, 2, 1]);
    dataIn(:, :, batchExtendLen + 1 : end)    = wiredOR;
    %% Pre Detection
    for t = batchExtendLen + 1 : size(dataIn, 3)
        nonCollisionCntr(nonCollisionCntr < nonCollisionCntrH & squeeze(dataIn(:, :, t)) ~= 0) = nonCollisionCntr(nonCollisionCntr < nonCollisionCntrH & squeeze(dataIn(:, :, t)) ~= 0) + 1;
        nonCollisionCntr(nonCollisionCntr > 0 & squeeze(dataIn(:, :, t)) == 0) = nonCollisionCntr(nonCollisionCntr > 0 & squeeze(dataIn(:, :, t)) == 0) - 1;
        
        preDetectionTemp = (nonCollisionCntr >= preDetectionThr) & squeeze(dataIn(:, :, t)) ~= 0;
        preDetectedSpike(:, :, (b - 1) * batchSize + (t - batchExtendLen)) = preDetectionTemp;
    end
    %% minimumFinder
    detectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize) = preDetectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize) & (dataIn(:, :, batchExtendLen + 1 - 2 : end - 2) < dataIn(:, :, batchExtendLen + 1 - 3 : end - 3)) & (dataIn(:, :, batchExtendLen + 1 - 2 : end - 2) <= dataIn(:, :, batchExtendLen + 1 - 1 : end - 1)) & (dataIn(:, :, batchExtendLen + 1 - 2 : end - 2) <= dataIn(:, :, batchExtendLen + 1 : end));
    %% Amplitude Threshold
    detectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize) =  detectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize) & (dataIn(:, :, batchExtendLen + 1 - 2 : end - 2)) < (-1 * ampThreshold);
    %% NEO Threshold
    detectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize)    =  detectedSpike(:, :, (b - 1) * batchSize + 1 : b * batchSize) & abs(double(dataIn(:, :, batchExtendLen + 1 - 2 : end - 2)) .* double(dataIn(:, :, batchExtendLen + 1 - 2 : end - 2)) - double(dataIn(:, :, batchExtendLen + 1 - 1 : end - 1)) .* double(dataIn(:, :, batchExtendLen + 1 - 3 : end - 3))) > NEOThreshold;

    %% Central |Channel Check
    if centralChannelDetection == 1
        for r = 1 : RowNum
            for c = 1: ColNum
                [NeighborElecsRow, NeighborElecsCol] = FindNeighborElec(r, c, RowNum, ColNum);
                detIndex = find(squeeze(detectedSpike(r, c, (b - 1) * batchSize + 1 : b * batchSize)) == 1);
                for t = 1 : numel(detIndex)
                    ind0 = sub2ind([RowNum, ColNum, batchSize + batchExtendLen], r, c, batchExtendLen + detIndex(t) - 2);

                    ind1 = sub2ind([RowNum, ColNum, batchSize + batchExtendLen], NeighborElecsRow, NeighborElecsCol, batchExtendLen + detIndex(t) - 0);
                    ind2 = sub2ind([RowNum, ColNum, batchSize + batchExtendLen], NeighborElecsRow, NeighborElecsCol, batchExtendLen + detIndex(t) - 1);
                    ind3 = sub2ind([RowNum, ColNum, batchSize + batchExtendLen], NeighborElecsRow, NeighborElecsCol, batchExtendLen + detIndex(t) - 2);
                    ind4 = sub2ind([RowNum, ColNum, batchSize + batchExtendLen], NeighborElecsRow, NeighborElecsCol, batchExtendLen + detIndex(t) - 3);
                
                    mainCh       = dataIn(r, c, batchExtendLen + detIndex(t) - 3 : batchExtendLen + detIndex(t));
                    MainMin      = abs(min(min(dataIn(ind0))));
                    Neighbors    = [dataIn(ind1); dataIn(ind2); dataIn(ind3); dataIn(ind4)];
                    NeighborsMin = abs(min(min(Neighbors)));


                    if NeighborsMin > MainMin
                        detectedSpike(r, c, (b - 1) * batchSize + detIndex(t)) = 0;
                    end
                end
            end
        end
    end
end
detectedSpike(:, :, 1 : end - 2)      = detectedSpike(:, :, 3 : end);
detectedSpike(:, :, end - 1 : end)    = 0;   
preDetectedSpike(:, :, 1 : end - 2)   = preDetectedSpike(:, :, 3 : end);
preDetectedSpike(:, :, end - 1 : end) = 0;


save(strcat(outputPath, '\', 'Detection_', num2str(ampThreshold), '_', num2str(NEOThreshold),'.mat'), 'detectedSpike', 'preDetectedSpike', 'Fs', 'RowNum', 'ColNum', 'startTime', 'sampleNum','preDetectionThr','ampThreshold', 'NEOThreshold', '-v7.3');