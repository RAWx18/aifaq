import React, { useRef, useEffect } from 'react';
import { Message } from '@/lib/types';
import ChatRequest from './chat-request';
import ChatResponse from './chat-response';
import ThinkingAnimation from './thinking-animation';

interface ChatSectionProps {
	messages: Message[];
}

const ChatSection: React.FC<ChatSectionProps> = ({
	messages
}) => {
	const chatContainerRef =
		useRef<HTMLDivElement>(null);
	const lastMessageIsUser =
		messages.length > 0 &&
		messages[messages.length - 1].type === 0;

	useEffect(() => {
		if (chatContainerRef.current) {
			chatContainerRef.current.scrollTop =
				chatContainerRef.current.scrollHeight;
		}
	}, [messages]);

	return (
		<div className="relative flex flex-col items-center py-2 h-full overflow-y-auto">
			<div
				className="flex flex-col max-w-3xl w-full flex-grow space-y-2"
				ref={chatContainerRef}
			>
				{messages.map((message, index) => (
					<div key={message.id}>
						{message.type === 0 ? (
							<ChatRequest
								request={message.content}
							/>
						) : (
							<ChatResponse message={message} />
						)}
					</div>
				))}

				{/* Show thinking animation when last message is from user */}
				{lastMessageIsUser && (
					<div className="flex flex-row space-x-4 p-4">
						<div className="h-10 w-10 rounded-full border shrink-0 p-2 bg-background flex items-center justify-center">
							<ThinkingAnimation />
						</div>
						<div className="flex items-center">
							<p className="text-muted-foreground">
								AIFAQ is thinking...
							</p>
						</div>
					</div>
				)}
			</div>
		</div>
	);
};

export default ChatSection;
