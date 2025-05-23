import { SERVER_BASE_URL } from '../const';
import { Message } from '@/lib/types';
import { handleMultiAgentSend } from './multiAgentFetcher';

// Flag to enable multi-agent mode
export const USE_MULTI_AGENT = true;

export async function handleSend(
	message: Message
) {
	// If multi-agent mode is enabled, use the multi-agent endpoint
	if (USE_MULTI_AGENT) {
		return handleMultiAgentSend(message);
	}

	// Otherwise use the original endpoint
	return fetch(SERVER_BASE_URL + '/query', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(message)
	})
		.then((response) => {
			return response.json();
		})
		.catch((err) => {
			console.log(err);
		});
}
