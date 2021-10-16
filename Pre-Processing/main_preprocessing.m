clc
close all
clear all


mkdir('Risultati');
S=dir(fullfile('old','*.mat'));

% Preprocess all the data files in the folder and saves in Results

for c=1:size(S)
    
    filename=S(c).name;
    load(filename);
    
    fs = 256; % sampling frequency
    x = y(2:9,:)'; % data matrix (channel x AllTrial)
    trig=y(10,:); % Trigger vector
    
    N=size(x,1); % number of trials
    t=y(1,:); % time vector 
    epoch=cell(2,1); 

    %variable initialization
    latency=[];
    type=[];
    duration=[];

    %find the start and the end of each epoch
    idx=find(diff(trig)~=0);
    a=0; % the trials are ordered as explained in classInfo_4_5.m
    
    % The first corresponds to 9Hz (a=0)
    % The second to 10Hz (a=1)
    % The third to 12Hz (a=2)
    % The fourth to 15Hz (a=3)
     
    for i=2:length(trig)-1
        if a==4
            a=0;
        end
        if trig(i)==1 && trig(i-1)==0
            latency=[latency;i];
            type=[type;a];
            a=a+1;
        elseif trig(i)==0 && trig(i-1)==1
            duration=[duration;(i-latency(end))];
        end
    end

    trigger=table(type,latency,duration);
    
    % epochs division: [#epochs, #channels, time points]
    [URaw, RRaw, DRaw, LRaw]=epochsDivide(x',trigger);

    %Nyquist frequency
    fN=fs/2;
    
    %% TEMPORAL FILTERING %%
    
    % bandpass filtering for alpha-beta rythms, 8th order, butterworth
    [b,a]=butter(8,[8 30]/fN, 'bandpass');

    % actual filtering
    Y=filtfilt(b,a,x);
    
    % epochs division: [#epochs, #channels, time points]
    [UFilt, RFilt, DFilt, LFilt]=epochsDivide(Y',trigger);
    
    
    %% SPATIAL FILTERING %%
    
    Zica = fastICA(x',1);
    % epochs division: [#epochs, #channels, time points]
    [Uica, Rica, Dica, Lica]=epochsDivide(Zica,trigger);
    
    %% SAVING DATA %%
    
    save(fullfile(pwd,'Risultati',strcat('Processed_',filename)),'URaw', 'RRaw', 'DRaw', 'LRaw', 'UFilt', 'RFilt', 'DFilt', 'LFilt','Uica', 'Rica', 'Dica', 'Lica');
end