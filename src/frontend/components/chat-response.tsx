import React from 'react';
import { IconHyperledger } from './icons/IconHyperledger';
import ChatResponseOptions from './chat-response-options';
import MultiAgentInfo from './multi-agent-info';
import { Message } from '@/lib/types';

interface Props {
	message: Message;
}

const ChatResponse = ({ message }: Props) => (
	<div className="flex flex-row space-x-4 p-4">
		<div className="h-10 w-10 rounded-full border shrink-0 p-2 bg-background">
			<IconHyperledger />
		</div>

		<div className="flex flex-col">
			<div className="w-fit p-2">
				<p> {message.content} </p>
			</div>
			<div className="mt-4">
				<ChatResponseOptions
					text={message.content}
				/>
				{message.metadata && (
					<MultiAgentInfo
						metadata={message.metadata}
					/>
				)}
			</div>
		</div>
	</div>
);

export default ChatResponse;
