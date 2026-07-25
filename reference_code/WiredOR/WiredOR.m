function [dataOut] = WiredOR(dataIn, wireNum)

lsb               = 1;
dataOut           = zeros(size(dataIn));
ramp(1, 1, :)     = -(2 ^ ceil(log2(abs(min(min(min(dataIn))))))) : lsb : (2 ^ ceil(log2(abs(max(max(max(dataIn))))))) - 1;

%ramp(1, 1, 1 : 512) = -256 : lsb : 255; 

rampStepNum       = length(ramp);

for wire = 1 : wireNum
    dataInTemp    = dataIn(wire : wireNum : end, :, :);
    ramp          = repmat(ramp(1, 1, :), size(dataInTemp, 1), size(dataInTemp, 2), 1);
    rampNext      = ramp + lsb;
    
    for i = 1 : size(dataInTemp, 3)
        rawDataComp = squeeze(dataInTemp(:, :, i));
        rawDataComp = repmat(rawDataComp, 1, 1, rampStepNum);
        rampComp    = (rawDataComp >= ramp & rawDataComp < rampNext);
    
        wireOrOutR        = logical(squeeze(any(rampComp, 2)));
        wireOrOutC        = logical(squeeze(any(rampComp, 1)));
    
        collisionFree     = sum(squeeze(any(rampComp, 1)), 1) == 1 | sum(squeeze(any(rampComp, 2)), 1) == 1;
        for row = wire : wireNum : size(dataIn, 1)
            for col = 1 : size(dataInTemp, 2)
                dataOut(row, col, i) = dataIn(row, col, i) * any(wireOrOutR(floor((row - 1) / wireNum) + 1, collisionFree) .* wireOrOutC(col, collisionFree));
            end
        end
    end
end