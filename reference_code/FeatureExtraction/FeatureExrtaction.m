
clear all
clc
%% Load Files 
[fileDet,pathDet]   = uigetfile('../Database/*.mat','Select Detection File');
load(strcat(pathDet, fileDet), 'detectedSpike', 'ampThreshold', 'NEOThreshold', 'sampleNum');

[file,path]         = uigetfile('../Database/*.h5','Select Wired-OR File');
recordingFile       = strcat(path, file);
%% Parameters
RowNum              = size(detectedSpike, 1);
ColNum              = size(detectedSpike, 2);

features            = cell(RowNum, ColNum);
sampleNum           = size(detectedSpike, 3);

sinCosWidth         = 4;
dividerWidth        = 12;
spatialResolution   = 5;
batchLen            = 20000;

spikeCntr = 0;
for row = 1 : RowNum
    for col = 1 : ColNum
        features{row, col}.X             = [];
        features{row, col}.Y             = [];
        features{row, col}.time          = [];
    end
end
%% Spike Detection Process
for b = 1 : batchLen : sampleNum
    disp(strcat(num2str(b), '/', num2str(sampleNum)))

    if b == 1 
        wiredORData = zeros(batchLen + 3, ColNum, RowNum);
        wiredORData(2 : end, :, :) = double(h5read(recordingFile,'/wiredOR', [b, 1, 1], [batchLen + 2, ColNum, RowNum]));
    elseif b + batchLen + 2 > sampleNum
        batcheLen   = sampleNum - b + 1;
        wiredORData = zeros(batchLen + 3, ColNum, RowNum);
        wiredORData(1 : end - 2, :, :) = double(h5read(recordingFile,'/wiredOR', [b - 1, 1, 1], [batchLen + 1, ColNum, RowNum]));
    else
        wiredORData = double(h5read(recordingFile,'/wiredOR', [b - 1, 1, 1], [batchLen + 3, ColNum, RowNum]));
    end

    for row = 1 : RowNum
        for col = 1 : ColNum    
            [NeighborElecsRow, NeighborElecsCol, NeighborElecsCos, NeighborElecsSin, WrongPosition] = FindNeighborElec(row, col, RowNum, ColNum);
            NeighborElecsCosQ  = round(NeighborElecsCos * 2 ^ sinCosWidth);
            NeighborElecsSinQ  = round(NeighborElecsSin * 2 ^ sinCosWidth);

            spikeTimes         = find(squeeze(detectedSpike(row, col, b : b + batchLen - 1) == 1)) + 1;
            if ~isempty(spikeTimes)
                mainChannel        = double(wiredORData(:, col, row)).';
                mainChannel        = [mainChannel(:, spikeTimes - 1); mainChannel(:, spikeTimes); mainChannel(:, spikeTimes + 1); mainChannel(:, spikeTimes + 2)];
                NeighborChannels   = zeros(numel(NeighborElecsRow), 4, numel(spikeTimes));
                for n = 1 : numel(NeighborElecsRow)
                    NeighborChannel          = double(wiredORData(:, NeighborElecsCol(n), NeighborElecsRow(n))).';
                    NeighborChannels(n, :, :)= WrongPosition(n) * [NeighborChannel(:, spikeTimes - 1); NeighborChannel(:, spikeTimes); NeighborChannel(:, spikeTimes + 1); NeighborChannel(:, spikeTimes + 2)];
                end
        
                mainChannelAmp      = abs(min(mainChannel, [], 1));
                NeighborChannelsAmp = abs(squeeze(min(NeighborChannels, [], 2)));
                mainNeighborAmp     = abs(max(NeighborChannelsAmp, [], 1));
    
                spatialVector       = round((repmat(NeighborElecsCosQ.', 1, numel(spikeTimes)) .* NeighborChannelsAmp + 1i * repmat(NeighborElecsSinQ.', 1, numel(spikeTimes)) .* NeighborChannelsAmp)/ (2 ^ sinCosWidth));
                spatialVector       = sum(spatialVector, 1);

                spatialVectorAbs        = max([abs(real(spatialVector)); abs(imag(spatialVector))], [], 1) + round(min([abs(real(spatialVector)); abs(imag(spatialVector))], [], 1) / 2);
                AmpCoef                 = round((double(mainNeighborAmp * (2 ^ dividerWidth)) ./ (double(mainChannelAmp) .* spatialVectorAbs)));
                
                AmpCoef(mainNeighborAmp == 0)       = 0;
                mainNeighborAmp(mainChannelAmp == 0) = [];
                spikeTimes(mainChannelAmp == 0)      = [];
                spatialVector(mainChannelAmp == 0)   = [];
                mainChannelAmp(mainChannelAmp == 0)  = [];
        

                features{row, col}.X             = [features{row, col}.X, round(real(spatialVector) .* AmpCoef / (2 ^ (dividerWidth - spatialResolution)))];
                features{row, col}.Y             = [features{row, col}.Y, round(imag(spatialVector) .* AmpCoef / (2 ^ (dividerWidth - spatialResolution)))];
                features{row, col}.time          = [features{row, col}.time, (spikeTimes.' - 1 + b - 1)];
        
                features{row, col}.X(isnan(features{row, col}.X)) = 0;
                features{row, col}.Y(isnan(features{row, col}.Y)) = 0;
    
                features{row, col}.X(features{row, col}.X > 31)  = 31;
                features{row, col}.X(features{row, col}.X < -31) = -31;
                features{row, col}.Y(features{row, col}.Y > 31)  = 31;
                features{row, col}.Y(features{row, col}.Y < -31) = -31;
            end
    
        end
    end
end
if strcmp(fileDet(1 : 11), 'DetectionGT')
    save(strcat(path, '\','FeaturesGT', fileDet(12 : end)), 'features', 'spatialResolution', 'sampleNum');
else
    save(strcat(path, '\','Features_', num2str(ampThreshold), '_', num2str(NEOThreshold),'.mat'), 'features', 'spatialResolution', 'sampleNum', 'ampThreshold', 'NEOThreshold');
end

