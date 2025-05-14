'use client';

import React, { useState } from 'react';
import { MultiAgentMetadata } from '@/lib/types/multiAgentTypes';
import { Button } from './ui/button';
import { Info } from 'lucide-react';

interface Props {
	metadata?: MultiAgentMetadata;
}

const MultiAgentInfo = ({ metadata }: Props) => {
	const [isOpen, setIsOpen] = useState(false);

	if (!metadata) return null;

	const toggleOpen = () => {
		setIsOpen(!isOpen);
	};

	// Format evaluation scores to be more readable
	const formatScore = (score?: number) => {
		if (score === undefined) return 'N/A';
		return (score * 100).toFixed(0) + '%';
	};

	return (
		<div className="mt-2">
			<Button
				variant="ghost"
				size="sm"
				onClick={toggleOpen}
				className="flex items-center gap-1 text-xs text-muted-foreground"
			>
				<Info className="h-3 w-3" />
				{isOpen
					? 'Hide AI analysis'
					: 'Show AI analysis'}
			</Button>

			{isOpen && (
				<div className="mt-2 p-3 bg-muted/50 rounded-md text-xs">
					<div className="grid grid-cols-1 md:grid-cols-2 gap-2">
						<div>
							<h4 className="font-medium mb-1">
								Query Understanding
							</h4>
							<p>
								Type:{' '}
								<span className="font-medium">
									{
										metadata.query_understanding
											.query_type
									}
								</span>
							</p>
							<p>
								Key terms:{' '}
								{metadata.query_understanding.key_terms.join(
									', '
								)}
							</p>
						</div>

						<div>
							<h4 className="font-medium mb-1">
								Retrieval
							</h4>
							<p>
								Documents retrieved:{' '}
								{
									metadata.retrieval
										.document_count
								}
							</p>
						</div>

						<div>
							<h4 className="font-medium mb-1">
								Evaluation
							</h4>
							<div className="grid grid-cols-2 gap-1">
								<p>
									Relevance:{' '}
									{formatScore(
										metadata.evaluation.relevance
									)}
								</p>
								<p>
									Grounding:{' '}
									{formatScore(
										metadata.evaluation.grounding
									)}
								</p>
								<p>
									Completeness:{' '}
									{formatScore(
										metadata.evaluation
											.completeness
									)}
								</p>
								<p>
									Coherence:{' '}
									{formatScore(
										metadata.evaluation.coherence
									)}
								</p>
								<p>
									Conciseness:{' '}
									{formatScore(
										metadata.evaluation
											.conciseness
									)}
								</p>
							</div>
						</div>

						<div>
							<h4 className="font-medium mb-1">
								Processing Pipeline
							</h4>
							<p>
								{metadata.processing_stages.join(
									' → '
								)}
							</p>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};

export default MultiAgentInfo;
