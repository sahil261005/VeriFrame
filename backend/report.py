import os

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception as err:
    print(f"warning: weasyprint could not be imported: {err}. falling back to plain html report.")
    WEASYPRINT_AVAILABLE = False


def generate_pdf(report_dict: dict) -> bytes:
    """
    generates a PDF report from a report dictionary.
    uses WeasyPrint if available, otherwise returns HTML as bytes.
    """
    
    # build custom HTML template with inline styling
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>VeriFrame Analysis Report</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #333;
                margin: 40px;
                line-height: 1.6;
            }}
            .header {{
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                color: #1e3a8a;
                font-size: 28px;
            }}
            .verdict-badge {{
                display: inline-block;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 4px;
                color: #fff;
                margin-top: 15px;
            }}
            .badge-manipulated {{
                background-color: #ef4444;
            }}
            .badge-authentic {{
                background-color: #10b981;
            }}
            .badge-uncertain {{
                background-color: #f59e0b;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .section-title {{
                font-size: 20px;
                color: #1e3a8a;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 5px;
                margin-bottom: 15px;
            }}
            .meta-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .meta-table td, .meta-table th {{
                padding: 8px 12px;
                border: 1px solid #e5e7eb;
                text-align: left;
            }}
            .meta-table th {{
                background-color: #f9fafb;
                font-weight: bold;
                width: 30%;
            }}
            .agent-card {{
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #f9fafb;
            }}
            .agent-name {{
                font-weight: bold;
                font-size: 16px;
                color: #2563eb;
                margin-bottom: 5px;
            }}
            .explanation-item {{
                margin-bottom: 15px;
                padding-left: 15px;
                border-left: 3px solid #3b82f6;
            }}
            .timestamp {{
                font-weight: bold;
                color: #4b5563;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>VeriFrame Analysis Report</h1>
            <p>Video Analysis File: {report_dict.get('video_metadata', {}).get('filename', 'N/A')}</p>
            
            <div class="verdict-badge badge-{report_dict.get('verdict', 'UNCERTAIN').lower()}">
                VERDICT: {report_dict.get('verdict', 'UNCERTAIN')} ({(report_dict.get('confidence', 0.0) * 100):.1f}% confidence)
            </div>
        </div>

        <div class="section">
            <div class="section-title">Video Metadata</div>
            <table class="meta-table">
                <tr>
                    <th>Duration</th>
                    <td>{report_dict.get('video_metadata', {}).get('duration', 0.0):.2f} seconds</td>
                </tr>
                <tr>
                    <th>Resolution</th>
                    <td>{report_dict.get('video_metadata', {}).get('width', 0)}x{report_dict.get('video_metadata', {}).get('height', 0)}</td>
                </tr>
                <tr>
                    <th>FPS / Frame Count</th>
                    <td>{report_dict.get('video_metadata', {}).get('fps', 0.0):.1f} / {report_dict.get('video_metadata', {}).get('frame_count', 0)} frames</td>
                </tr>
                <tr>
                    <th>Partial Analysis (Failure Fallback)</th>
                    <td>{"Yes (some agents failed to run)" if report_dict.get('is_partial_analysis', False) else "No (all agents ran successfully)"}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <div class="section-title">Agent Analysis Breakdown</div>
            
            <div class="agent-card">
                <div class="agent-name">Visual Forensics Agent</div>
                <div>Status: {report_dict.get('agent_breakdown', {}).get('visual_agent', {}).get('status', 'unknown')}</div>
                <div>Deepfake Score: {report_dict.get('agent_breakdown', {}).get('visual_agent', {}).get('score', 0.0):.2f}</div>
                <div>Flagged Frames Count: {report_dict.get('agent_breakdown', {}).get('visual_agent', {}).get('flagged_frames_count', 0)}</div>
            </div>

            <div class="agent-card">
                <div class="agent-name">Temporal Consistency Agent</div>
                <div>Status: {report_dict.get('agent_breakdown', {}).get('temporal_agent', {}).get('status', 'unknown')}</div>
                <div>Inconsistency Score: {report_dict.get('agent_breakdown', {}).get('temporal_agent', {}).get('score', 0.0):.2f}</div>
                <div>Optical Flow Spikes: {report_dict.get('agent_breakdown', {}).get('temporal_agent', {}).get('flow_anomalies_count', 0)}</div>
                <div>Facial Mesh Inconsistencies: {report_dict.get('agent_breakdown', {}).get('temporal_agent', {}).get('face_inconsistencies_count', 0)}</div>
            </div>

            <div class="agent-card">
                <div class="agent-name">LLM Reasoning Agent</div>
                <div>Status: {report_dict.get('agent_breakdown', {}).get('llm_agent', {}).get('status', 'unknown')}</div>
                <div>Average Fake Rating: {report_dict.get('agent_breakdown', {}).get('llm_agent', {}).get('score', 0.0):.2f}</div>
                <div>Summary Assessment: {report_dict.get('agent_breakdown', {}).get('llm_agent', {}).get('reasoning', 'N/A')}</div>
            </div>
        </div>
    """

    # add frame explanations if available
    frame_details = report_dict.get("frame_level_details", {})
    if frame_details:
        html_content += """
        <div class="section">
            <div class="section-title">Frame-by-Frame Forensics (LLM Explanations)</div>
        """
        for ts, explanation in frame_details.items():
            html_content += f"""
            <div class="explanation-item">
                <div class="timestamp">Frame Timestamp: {ts}s</div>
                <div>{explanation}</div>
            </div>
            """
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    # convert to PDF bytes if weasyprint works, otherwise return HTML text bytes
    if WEASYPRINT_AVAILABLE:
        try:
            pdf_bytes = HTML(string=html_content).write_pdf()
            return pdf_bytes
        except Exception as e:
            print(f"error rendering PDF with weasyprint: {e}. returning raw HTML instead.")
            return html_content.encode("utf-8")
    else:
        # fallback to plain HTML bytes
        return html_content.encode("utf-8")
