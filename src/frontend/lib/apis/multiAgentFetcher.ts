import { SERVER_BASE_URL } from '../const';
import { Message } from '@/lib/types';

// Updated interface to include metadata from multi-agent response
export interface MultiAgentResponse {
	id: string;
	message: Message;
	metadata: {
		query_understanding: {
			query_type: string;
			key_terms: string[];
		};
		retrieval: {
			document_count: number;
		};
		evaluation: Record<string, number>;
		processing_stages: string[];
	};
}

export async function handleMultiAgentSend(
	message: Message
): Promise<MultiAgentResponse | undefined> {
	return fetch(
		SERVER_BASE_URL + '/query/multi-agent',
		{
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				id: message.id,
				content: message.content
			})
		}
	)
		.then((response) => {
			return response.json();
		})
		.catch((err) => {
			console.log(err);
		});
}
