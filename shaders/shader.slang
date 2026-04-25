struct VSInput
{
	float2 pos;
	float3 col;
};

struct VSOutput
{
	float4 pos : SV_Position;
	float3 col;
}

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
	VSOutput output;
	output.pos = float4(input.pos, 0.0, 1.0);
	output.col = input.col;
	return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target
{
	return float4(vertIn.col, 1.0);
}
