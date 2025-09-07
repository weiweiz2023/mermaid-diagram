```mermaid

graph TB
    %% Input Layer
    Input["Original Input<br/>[batch, 2304_features, patches]<br/>Example: 256×3²=2304 features"]
    Weight["Original Weight<br/>[out_ch, 2304_features]"]
    
    %% BitStream Processing
    subgraph "BitStream Processing (Input Quantization)"
        InputQuant["VectorizedInputBitStreaming<br/>4bit → 4 streams<br/>[batch, 2304, patches, 4_streams]"]
        Stream0["Stream 0 (bit0)<br/>weight=2^0=1"]
        Stream1["Stream 1 (bit1)<br/>weight=2^1=2"] 
        Stream2["Stream 2 (bit2)<br/>weight=2^2=4"]
        Stream3["Stream 3 (bit3)<br/>weight=2^3=8"]
    end
    
    %% BitSlice Processing
    subgraph "BitSlice Processing (Weight Quantization)"
        WeightQuant["VectorizedWeightBitSlicing<br/>4bit → 4 slices<br/>[out_ch, 2304, 4_slices]"]
        Slice0["Slice 0 (bit0)<br/>weight=2^0=1"]
        Slice1["Slice 1 (bit1)<br/>weight=2^1=2"]
        Slice2["Slice 2 (bit2)<br/>weight=2^2=4"] 
        Slice3["Slice 3 (bit3)<br/>weight=2^3=8"]
    end
    
    %% Chunk Division (Subarray Processing)
    subgraph "Chunk Division (subarray_size=128)"
        ChunkSplit["torch.chunk(dim=1)<br/>2304 → 18 chunks<br/>17×128 + 1×96"]
        
        InputChunk0["Input Chunk 0<br/>[batch, 128, patches, 4]"]
        InputChunk1["Input Chunk 1<br/>[batch, 128, patches, 4]"]
        InputChunkN["Input Chunk 17<br/>[batch, 96, patches, 4]"]
        
        WeightChunk0["Weight Chunk 0<br/>[out_ch, 128, 4]"]
        WeightChunk1["Weight Chunk 1<br/>[out_ch, 128, 4]"]
        WeightChunkN["Weight Chunk 17<br/>[out_ch, 96, 4]"]
    end
    
    %% Einstein Summation (16 combinations per chunk)
    subgraph "Einstein Summation (Per Chunk)"
        Einstein0["Chunk 0: Einstein Sum<br/>torch.einsum('bfps,oft->bopst')<br/>[batch, out_ch, patches, 4, 4]<br/>16 independent combinations"]
        Einstein1["Chunk 1: Einstein Sum<br/>[batch, out_ch, patches, 4, 4]<br/>16 independent combinations"]
        EinsteinN["Chunk 17: Einstein Sum<br/>[batch, out_ch, patches, 4, 4]<br/>16 independent combinations"]
    end
    
    %% Shared ADC Processing
    subgraph "Shared ADC Processing"
        ADCShared["Shared ADC Instance<br/>self.adc_pos & self.adc_neg<br/>All chunks use same parameters<br/>step_size, zero_point"]
        
        Flatten0["Chunk 0: flatten<br/>[batch×out_ch×patches×16]"]
        Flatten1["Chunk 1: flatten<br/>[batch×out_ch×patches×16]"]
        FlattenN["Chunk 17: flatten<br/>[batch×out_ch×patches×16]"]
        
        Quantized0["Chunk 0: Quantized Result<br/>view back to [batch,out_ch,patches,4,4]"]
        Quantized1["Chunk 1: Quantized Result<br/>view back to [batch,out_ch,patches,4,4]"]
        QuantizedN["Chunk 17: Quantized Result<br/>view back to [batch,out_ch,patches,4,4]"]
    end
    
    %% Weight Scaling and Aggregation
    subgraph "Weight Scaling & Aggregation"
        Scale0["Chunk 0: Scaling<br/>×stream_weights×slice_weights<br/>16 combinations scaled individually"]
        Scale1["Chunk 1: Scaling<br/>×stream_weights×slice_weights"]
        ScaleN["Chunk 17: Scaling<br/>×stream_weights×slice_weights"]
        
        Sum0["Chunk 0: sum(dim=(-2,-1))<br/>[batch, out_ch, patches]"]
        Sum1["Chunk 1: sum(dim=(-2,-1))<br/>[batch, out_ch, patches]"]
        SumN["Chunk 17: sum(dim=(-2,-1))<br/>[batch, out_ch, patches]"]
    end
    
    %% Final Accumulation
    FinalSum["torch.stack().sum(dim=0)<br/>Accumulate all chunk results<br/>[batch, out_ch, patches]"]
    
    %% Forward Propagation Connections
    Input --> InputQuant
    Weight --> WeightQuant
    
    InputQuant --> Stream0
    InputQuant --> Stream1
    InputQuant --> Stream2
    InputQuant --> Stream3
    
    WeightQuant --> Slice0
    WeightQuant --> Slice1
    WeightQuant --> Slice2
    WeightQuant --> Slice3
    
    Stream0 --> ChunkSplit
    Stream1 --> ChunkSplit
    Stream2 --> ChunkSplit
    Stream3 --> ChunkSplit
    
    Slice0 --> ChunkSplit
    Slice1 --> ChunkSplit
    Slice2 --> ChunkSplit
    Slice3 --> ChunkSplit
    
    ChunkSplit --> InputChunk0
    ChunkSplit --> InputChunk1
    ChunkSplit --> InputChunkN
    ChunkSplit --> WeightChunk0
    ChunkSplit --> WeightChunk1
    ChunkSplit --> WeightChunkN
    
    InputChunk0 --> Einstein0
    WeightChunk0 --> Einstein0
    InputChunk1 --> Einstein1
    WeightChunk1 --> Einstein1
    InputChunkN --> EinsteinN
    WeightChunkN --> EinsteinN
    
    Einstein0 --> Flatten0
    Einstein1 --> Flatten1
    EinsteinN --> FlattenN
    
    Flatten0 --> ADCShared
    Flatten1 --> ADCShared
    FlattenN --> ADCShared
    
    ADCShared --> Quantized0
    ADCShared --> Quantized1
    ADCShared --> QuantizedN
    
    Quantized0 --> Scale0
    Quantized1 --> Scale1
    QuantizedN --> ScaleN
    
    Scale0 --> Sum0
    Scale1 --> Sum1
    ScaleN --> SumN
    
    Sum0 --> FinalSum
    Sum1 --> FinalSum
    SumN --> FinalSum
    
    %% Backward Propagation Path
    subgraph "Backward Propagation Gradient Flow"
        Loss["∂Loss/∂output"]
        
        %% Gradient Distribution
        GradChunk0["∂Loss/∂chunk0_result<br/>= ∂Loss/∂output"]
        GradChunk1["∂Loss/∂chunk1_result<br/>= ∂Loss/∂output"]
        GradChunkN["∂Loss/∂chunkN_result<br/>= ∂Loss/∂output"]
        
        %% ADC Backward Propagation
        GradADC0["ADC Backward: chunk0<br/>STE + gradientFilter<br/>Using shared parameters"]
        GradADC1["ADC Backward: chunk1<br/>STE + gradientFilter<br/>Using shared parameters"]
        GradADCN["ADC Backward: chunkN<br/>STE + gradientFilter<br/>Using shared parameters"]
        
        %% Einstein Backward Propagation
        GradEin0["Einstein Backward: chunk0<br/>16 combinations → 4streams+4slices"]
        GradEin1["Einstein Backward: chunk1<br/>16 combinations → 4streams+4slices"]
        GradEinN["Einstein Backward: chunkN<br/>16 combinations → 4streams+4slices"]
        
        %% Gradient Recombination
        GradCombine["torch.cat(chunks, dim=1)<br/>Recombine features dimension<br/>128+128+...+96=2304"]
        
        %% BitSlice/Stream Backward
        GradBitStream["BitStream Backward<br/>mean(dim=-1): 4streams→1"]
        GradBitSlice["BitSlice Backward<br/>mean(dim=-1): 4slices→1"]
        
        %% Final Gradients
        GradInput["∂Loss/∂input<br/>[batch, 2304, patches]"]
        GradWeight["∂Loss/∂weight<br/>[out_ch, 2304]"]
    end
    
    %% Backward Propagation Connections
    Loss -.-> FinalSum
    FinalSum -.-> GradChunk0
    FinalSum -.-> GradChunk1
    FinalSum -.-> GradChunkN
    
    GradChunk0 -.-> GradADC0
    GradChunk1 -.-> GradADC1
    GradChunkN -.-> GradADCN
    
    GradADC0 -.-> GradEin0
    GradADC1 -.-> GradEin1
    GradADCN -.-> GradEinN
    
    GradEin0 -.-> GradCombine
    GradEin1 -.-> GradCombine
    GradEinN -.-> GradCombine
    
    GradCombine -.-> GradBitStream
    GradCombine -.-> GradBitSlice
    
    GradBitStream -.-> GradInput
    GradBitSlice -.-> GradWeight
    
    %% Key Statistics
    subgraph "Key Statistics"
        Stats["• Total Features: 2304<br/>• Chunks: 18 (17×128 + 1×96)<br/>• Combinations per chunk: 4×4=16<br/>• Total combinations: 18×16=288<br/>• ADC calls: 18×2=36 times<br/>• ADC instances: Only 2 (shared)<br/>• Gradient path: 288→36→18→1"]
    end
    
    %% Style Definitions
    classDef input fill:#e1f5fe
    classDef bitprocess fill:#f3e5f5
    classDef chunk fill:#fff3e0
    classDef einstein fill:#e8f5e8
    classDef adc fill:#ffecb3
    classDef gradient fill:#ffcdd2
    classDef stats fill:#f1f8e9
    
    class Input,Weight input
    class InputQuant,WeightQuant,Stream0,Stream1,Stream2,Stream3,Slice0,Slice1,Slice2,Slice3 bitprocess
    class ChunkSplit,InputChunk0,InputChunk1,InputChunkN,WeightChunk0,WeightChunk1,WeightChunkN chunk
    class Einstein0,Einstein1,EinsteinN,Scale0,Scale1,ScaleN,Sum0,Sum1,SumN,FinalSum einstein
    class ADCShared,Flatten0,Flatten1,FlattenN,Quantized0,Quantized1,QuantizedN adc
    class Loss,GradChunk0,GradChunk1,GradChunkN,GradADC0,GradADC1,GradADCN,GradEin0,GradEin1,GradEinN,GradCombine,GradBitStream,GradBitSlice,GradInput,GradWeight gradient
    class Stats stats
