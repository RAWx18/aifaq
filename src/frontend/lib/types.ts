export interface Message {
	/// 0: user | 1: agent
	type: 0 | 1;
	id: string;
	content: string;
	metadata?: any; // For multi-agent response metadata
}
