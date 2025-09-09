#!/usr/bin/env python3


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import subprocess
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
import json
import base64
from typing import Optional, Dict, List, Tuple

# Enhanced Visualization Classes
class UnifiedScoreCalculator:
    """Calculate unified confidence scores from multiple similarity metrics."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'column': 0.3, 'semantic': 0.5, 'example': 0.2}
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def normalize_semantic_score(self, semantic_score: float) -> float:
        """Convert semantic similarity from [-100,100] to [0,100] range."""
        return max(0, (semantic_score + 100) / 2)
    
    def handle_null_example_score(self, example_score) -> float:
        """Assign neutral score (50) when example match is unavailable."""
        return 50.0 if pd.isna(example_score) else float(example_score)
    
    def calculate_unified_score(self, row) -> float:
        """Calculate unified confidence score using weighted formula."""
        cns = float(row['column_name_similarity'])  # Column Name Similarity
        nss = self.normalize_semantic_score(float(row['semantic_similarity']))  # Normalized Semantic
        ems = self.handle_null_example_score(row['cell_match_score'])  # Example Match
        
        unified = (self.weights['column'] * cns + 
                  self.weights['semantic'] * nss + 
                  self.weights['example'] * ems)
        
        return round(unified, 2)

class EnhancedVisualization:
    """Create bar chart visualizations."""
    
    def __init__(self):
        pass
    
    def get_confidence_category(self, score: float) -> str:
        """Determine confidence category based on score."""
        if score >= 90:
            return 'high'
        elif score >= 70:
            return 'medium'
        else:
            return 'low'
    
    def apply_basic_categorization(self, df: pd.DataFrame, score_column: str) -> pd.DataFrame:
        """Apply basic categorization without color coding."""
        df_categorized = df.copy()
        df_categorized['confidence_category'] = df_categorized[score_column].apply(self.get_confidence_category)
        return df_categorized
    
    def create_score_breakdown_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing breakdown of all scores for top linkages."""
        # Sort by unified confidence and take top 20 results
        top_df = df.nlargest(20, 'unified_confidence')
        
        # Create linkage labels
        labels = [f"{row['source_sheet']}.{row['source_column']} → {row['target_sheet']}.{row['target_column']}"
                 for _, row in top_df.iterrows()]
        
        fig = go.Figure()
        
        # Add bars for each score type
        fig.add_trace(go.Bar(
            name='Column Name Similarity',
            x=labels,
            y=top_df['column_name_similarity']
        ))
        
        fig.add_trace(go.Bar(
            name='Semantic Similarity (Normalized)',
            x=labels,
            y=[max(0, (score + 100) / 2) for score in top_df['semantic_similarity']]
        ))
        
        fig.add_trace(go.Bar(
            name='Example Match Score',
            x=labels,
            y=[50 if pd.isna(score) else score for score in top_df['cell_match_score']]
        ))
        
        fig.add_trace(go.Bar(
            name='Unified Confidence',
            x=labels,
            y=top_df['unified_confidence']
        ))
        
        fig.update_layout(
            title='Score Breakdown Analysis - Top Linkages by Unified Confidence',
            xaxis_title='Column Linkages',
            yaxis_title='Score (0-100)',
            barmode='group',
            height=500,
            xaxis=dict(tickangle=45, tickfont=dict(size=10))
        )
        
        return fig
    
    def _get_connection_color(self, confidence: float) -> str:
        """Get optimized connection color based on confidence level."""
        if confidence >= 80:
            return 'rgba(34, 197, 94, 0.7)'  # Green for high confidence
        elif confidence >= 60:
            return 'rgba(59, 130, 246, 0.7)'  # Blue for medium confidence  
        elif confidence >= 40:
            return 'rgba(251, 191, 36, 0.7)'  # Yellow for low-medium confidence
        else:
            return 'rgba(239, 68, 68, 0.6)'  # Red for low confidence
    
    def _add_connection_lines_batch(self, fig: go.Figure, connections: list, table_positions: dict, table_width: int):
        """Add connection lines in batches for better performance."""
        # Group connections by confidence level for batch processing
        confidence_groups = {'high': [], 'medium': [], 'low': [], 'very_low': []}
        
        for conn in connections:
            confidence = conn['confidence']
            if confidence >= 80:
                confidence_groups['high'].append(conn)
            elif confidence >= 60:
                confidence_groups['medium'].append(conn)
            elif confidence >= 40:
                confidence_groups['low'].append(conn)
            else:
                confidence_groups['very_low'].append(conn)
        
        # Add connections by group for better rendering performance
        for group_name, group_connections in confidence_groups.items():
            if not group_connections:
                continue
                
            # Batch similar connections together
            x_coords = []
            y_coords = []
            hover_texts = []
            
            for conn in group_connections:
                source_pos = table_positions[conn['source_table']]
                target_pos = table_positions[conn['target_table']]
                
                # Calculate connection points
                source_x = source_pos['x'] + table_width // 2
                source_y = source_pos['y'] + 30
                target_x = target_pos['x'] + table_width // 2
                target_y = target_pos['y'] + 30
                
                x_coords.extend([source_x, target_x, None])  # None creates line breaks
                y_coords.extend([source_y, target_y, None])
                
                hover_text = f"<b style='color: white;'>🔗 Connection</b><br><span style='color: #e5e7eb;'>{conn['source_table']}.{conn['source_column']}</span><br><span style='color: #9ca3af;'>↓</span><br><span style='color: #e5e7eb;'>{conn['target_table']}.{conn['target_column']}</span><br><b style='color: #60a5fa;'>Confidence:</b> <span style='color: #fbbf24;'>{conn['confidence']:.1f}%</span>"
                hover_texts.extend([hover_text, hover_text, None])
            
            # Add single trace for all connections in this group
            if x_coords:
                sample_conn = group_connections[0]  # Use first connection for styling
                # Simplified line styling for better readability
                line_width = max(1, min(2.5, sample_conn['width'] * 0.7))  # Thinner lines
                line_opacity = max(0.2, min(0.6, sample_conn['opacity'] * 0.8))  # More subtle
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        color=sample_conn['color'],
                        width=line_width,
                        shape='linear',
                        dash='solid' if sample_conn['confidence'] >= 70 else 'dot'  # Dash low confidence
                    ),
                    opacity=line_opacity,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts,
                    showlegend=False,
                    name=f'{group_name}_connections'
                ))
    
    def _add_tables_batch(self, fig: go.Figure, tables: dict, table_positions: dict, table_width: int, max_columns_per_table: int = 15):
        """Add table boxes and columns with optimized batch rendering and lazy loading."""
        header_height = 50  # Increased for better readability
        column_height = 35  # Increased for better text visibility
        table_width = max(table_width, 200)  # Ensure minimum width for readability
        
        # Pre-calculate all table dimensions with column limiting
        table_heights = {}
        processed_tables = {}
        for table_name, columns in tables.items():
            # Limit columns for performance
            display_columns = columns[:max_columns_per_table]
            has_more = len(columns) > max_columns_per_table
            
            processed_tables[table_name] = {
                'columns': display_columns,
                'has_more': has_more,
                'total_columns': len(columns)
            }
            
            # Add extra height for "more" indicator if needed
            extra_height = column_height if has_more else 0
            table_heights[table_name] = header_height + len(display_columns) * column_height + 10 + extra_height
        
        # Batch add all table shapes first
        shapes = []
        annotations = []
        clickable_traces = []
        
        for table_name in processed_tables.keys():
            table_data = processed_tables[table_name]
            columns = table_data['columns']
            has_more = table_data['has_more']
            total_columns = table_data['total_columns']
            
            pos = table_positions[table_name]
            table_height = table_heights[table_name]
            
            # Add table shadow
            shapes.append(dict(
                type="rect",
                x0=pos['x'] + 3, y0=pos['y'] - 3,
                x1=pos['x'] + table_width + 3, y1=pos['y'] + table_height - 3,
                fillcolor="rgba(0, 0, 0, 0.15)",
                line=dict(width=0), layer="below"
            ))
            
            # Add table background
            shapes.append(dict(
                type="rect",
                x0=pos['x'], y0=pos['y'],
                x1=pos['x'] + table_width, y1=pos['y'] + table_height,
                fillcolor="rgba(20, 28, 48, 0.95)",
                line=dict(color="rgba(91, 194, 231, 0.6)", width=1.5),
                layer="below"
            ))
            
            # Add table header background
            shapes.append(dict(
                type="rect",
                x0=pos['x'], y0=pos['y'] + table_height - header_height,
                x1=pos['x'] + table_width, y1=pos['y'] + table_height,
                fillcolor="rgba(30, 58, 138, 0.9)",
                line=dict(color="rgba(91, 194, 231, 0.4)", width=0),
                layer="below"
            ))
            
            # Add table name annotation with column count
            table_display_name = f"{table_name} ({total_columns})" if has_more else table_name
            annotations.append(dict(
                x=pos['x'] + table_width // 2,
                y=pos['y'] + table_height - header_height // 2,
                text=f"<b style='font-weight: 700; letter-spacing: 0.02em; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);'>{table_display_name}</b>",
                showarrow=False,
                font=dict(
                    color="rgba(255, 255, 255, 1.0)", size=18,
                    family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                ),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"
            ))
            
            # Add column annotations and clickable areas
            column_x_coords = []
            column_y_coords = []
            column_names = []
            column_hover_texts = []
            
            for i, column in enumerate(columns):
                column_y = pos['y'] + table_height - header_height - (i + 1) * column_height + column_height // 2
                display_column = column if len(column) <= 25 else column[:22] + "..."
                
                # Add column name annotation
                annotations.append(dict(
                    x=pos['x'] + 12, y=column_y,
                    text=f"<span style='font-weight: 500; letter-spacing: 0.02em; text-shadow: 1px 1px 1px rgba(0,0,0,0.7);'>{display_column}</span>",
                    showarrow=False,
                    font=dict(
                        color="rgba(255, 255, 255, 1.0)", size=15,
                        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                    ),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    xanchor="left"
                ))
                
                # Collect clickable area data
                column_x_coords.append(pos['x'] + table_width // 2)
                column_y_coords.append(column_y)
                column_names.append(f"{table_name}.{column}")
                column_hover_texts.append(f"<b>{table_name}.{column}</b><br><i>Click to highlight related connections</i><br>Table: {table_name}")
            
            # Add "more columns" indicator if needed
            if has_more:
                more_y = pos['y'] + table_height - header_height - (len(columns) + 1) * column_height + column_height // 2
                annotations.append(dict(
                    x=pos['x'] + 12, y=more_y,
                    text=f"<span style='font-weight: 400; font-style: italic; letter-spacing: 0.02em; text-shadow: 1px 1px 1px rgba(0,0,0,0.6);'>... +{total_columns - len(columns)} more</span>",
                    showarrow=False,
                    font=dict(
                        color="rgba(200, 200, 200, 0.9)", size=13,
                        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                    ),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    xanchor="left"
                ))
            
            # Add single trace for all columns in this table
            if column_x_coords:
                clickable_traces.append(go.Scatter(
                    x=column_x_coords, y=column_y_coords,
                    mode='markers',
                    marker=dict(
                        size=20, color='rgba(91, 194, 231, 0.1)',
                        line=dict(color='rgba(91, 194, 231, 0.3)', width=1)
                    ),
                    hovertemplate='%{text}<extra></extra>',
                    text=column_hover_texts,
                    showlegend=False,
                    name=f"{table_name}_columns",
                    customdata=column_names,
                    textfont=dict(size=1, color='rgba(0,0,0,0)')
                ))
        
        # Batch add all shapes and annotations
        fig.update_layout(shapes=shapes, annotations=annotations)
        
        # Add all clickable traces
        for trace in clickable_traces:
            fig.add_trace(trace)
    
    def create_confidence_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing distribution of unified confidence scores."""
        # Create confidence ranges
        bins = list(range(0, 101, 10))
        labels = [f"{i}-{i+9}" for i in range(0, 90, 10)] + ["90-100"]
        df['confidence_range'] = pd.cut(df['unified_confidence'], bins=bins, labels=labels)
        
        range_counts = df['confidence_range'].value_counts().sort_index()
        
        # Filter out empty ranges (dynamic filtering)
        filtered_counts = range_counts[range_counts > 0]
        
        fig = go.Figure(data=[
            go.Bar(
                x=filtered_counts.index,
                y=filtered_counts.values,
                text=filtered_counts.values,
                textposition='auto',
                customdata=list(range(len(filtered_counts))),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        # Calculate appropriate y-axis range
        max_count = filtered_counts.max() if len(filtered_counts) > 0 else 10
        y_max = max(100, int(max_count * 1.1))  # Ensure minimum of 100 or 110% of max value
        
        fig.update_layout(
            title='Unified Confidence Score Distribution',
            xaxis_title='Confidence Range (%)',
            yaxis_title='Number of Linkages',
            height=400,
            yaxis=dict(
                range=[0, y_max],
                dtick=10 if y_max <= 100 else max(10, y_max // 10)
            )
        )
        
        return fig
    
    def create_semantic_similarity_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing semantic similarity distribution."""
        # Create semantic similarity ranges
        bins = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
        labels = ['-100 to -80', '-80 to -60', '-60 to -40', '-40 to -20', '-20 to 0',
                 '0 to 20', '20 to 40', '40 to 60', '60 to 80', '80 to 100']
        
        df['semantic_range'] = pd.cut(df['semantic_similarity'], bins=bins, labels=labels)
        range_counts = df['semantic_range'].value_counts().sort_index()
        
        # Filter out empty ranges (dynamic filtering)
        filtered_counts = range_counts[range_counts > 0]
        
        fig = go.Figure(data=[
            go.Bar(
                x=filtered_counts.index,
                y=filtered_counts.values,
                text=filtered_counts.values,
                textposition='auto',
                marker_color='lightblue',
                customdata=list(range(len(filtered_counts))),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Semantic Similarity Distribution',
            xaxis_title='Semantic Similarity Range',
            yaxis_title='Number of Linkages',
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_column_similarity_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bar chart showing column name similarity distribution."""
        # Create column similarity ranges
        bins = list(range(0, 101, 10))
        labels = [f"{i}-{i+9}" for i in range(0, 90, 10)] + ["90-100"]
        
        df['column_range'] = pd.cut(df['column_name_similarity'], bins=bins, labels=labels)
        range_counts = df['column_range'].value_counts().sort_index()
        
        # Filter out empty ranges (dynamic filtering)
        filtered_counts = range_counts[range_counts > 0]
        
        fig = go.Figure(data=[
            go.Bar(
                x=filtered_counts.index,
                y=filtered_counts.values,
                text=filtered_counts.values,
                textposition='auto',
                marker_color='lightgreen',
                customdata=list(range(len(filtered_counts))),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        # Calculate appropriate y-axis range
        max_count = filtered_counts.max() if len(filtered_counts) > 0 else 10
        y_max = max(100, int(max_count * 1.1))  # Ensure minimum of 100 or 110% of max value
        
        fig.update_layout(
            title='Column Name Similarity Distribution',
            xaxis_title='Column Similarity Range (%)',
            yaxis_title='Number of Linkages',
            height=400,
            yaxis=dict(
                range=[0, y_max],
                dtick=10 if y_max <= 100 else max(10, y_max // 10)
            )
        )
        
        return fig
    
    def create_er_diagram(self, df: pd.DataFrame) -> go.Figure:
        """Create an interactive ER diagram showing table relationships with optimized performance."""
        import math
        from collections import defaultdict
        
        # Calculate unified confidence scores if not present (optimized)
        df_with_confidence = df.copy()
        if 'unified_confidence' not in df_with_confidence.columns:
            calculator = UnifiedScoreCalculator()
            df_with_confidence['unified_confidence'] = df_with_confidence.apply(calculator.calculate_unified_score, axis=1)
        
        # Limit dataset size for performance (show top connections by confidence)
        max_connections = 500  # Configurable limit
        if len(df_with_confidence) > max_connections:
            df_with_confidence = df_with_confidence.nlargest(max_connections, 'unified_confidence')
            st.info(f"📊 Showing top {max_connections} connections by confidence for optimal performance. Total available: {len(df)}")
        
        # Optimized data structure building using defaultdict
        tables = defaultdict(set)
        connections = []
        
        # Vectorized data extraction for better performance
        for _, row in df_with_confidence.iterrows():
            source_table = row['source_sheet']
            target_table = row['target_sheet']
            source_column = row['source_column']
            target_column = row['target_column']
            confidence = row.get('unified_confidence', 0)
            
            # Add columns to tables
            tables[source_table].add(source_column)
            tables[target_table].add(target_column)
            
            # Store connection info with pre-calculated values
            connections.append({
                'source_table': source_table,
                'source_column': source_column,
                'target_table': target_table,
                'target_column': target_column,
                'confidence': confidence,
                'color': self._get_connection_color(confidence),
                'width': max(1.5, min(4, confidence / 20)),
                'opacity': max(0.3, min(0.8, confidence / 100))
            })
        
        # Convert to sorted lists once
        tables = {table: sorted(list(columns)) for table, columns in tables.items()}
        
        # Optimized layout calculation
        table_names = list(tables.keys())
        num_tables = len(table_names)
        
        # Improved grid layout with better spacing
        cols = min(3, math.ceil(math.sqrt(num_tables)))  # Limit columns for better readability
        rows = math.ceil(num_tables / cols)
        
        table_positions = {}
        table_width = 220  # Increased width for better text display
        table_spacing_x = 350  # Increased spacing to reduce clutter
        table_spacing_y = 280  # Increased vertical spacing
        
        # Pre-calculate all positions
        for i, table_name in enumerate(table_names):
            row = i // cols
            col = i % cols
            x = col * table_spacing_x + 80
            y = row * table_spacing_y + 80
            table_positions[table_name] = {'x': x, 'y': y}
        
        # Create figure with optimized rendering
        fig = go.Figure()
        
        # Batch connection lines for better performance
        self._add_connection_lines_batch(fig, connections, table_positions, table_width)
        
        # Add table boxes and columns with optimized rendering and lazy loading
        max_columns_display = 12 if len(tables) > 10 else 15  # Reduce columns for complex diagrams
        self._add_tables_batch(fig, tables, table_positions, table_width, max_columns_display)
        
        # Calculate figure dimensions
        max_x = max(pos['x'] for pos in table_positions.values()) + table_width + 50
        max_y = max(pos['y'] for pos in table_positions.values()) + 300  # Approximate max table height
        
        # Update layout with modern responsive design
        fig.update_layout(
            title=dict(
                text="<b style='font-weight: 600; letter-spacing: -0.02em;'>Entity Relationship Diagram</b>",
                font=dict(size=24, color="rgba(255, 255, 255, 0.95)", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                range=[-50, max_x],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                fixedrange=False,  # Allow zooming for better responsiveness
                showspikes=False
            ),
            yaxis=dict(
                range=[-50, max_y],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                fixedrange=False,  # Allow zooming for better responsiveness
                scaleanchor="x",
                scaleratio=1,
                showspikes=False
            ),
            plot_bgcolor="rgba(8, 12, 20, 1)",  # Darker, more modern background
            paper_bgcolor="rgba(8, 12, 20, 1)",
            font=dict(
                color="rgba(255, 255, 255, 0.9)", 
                family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                size=12
            ),
            height=max(650, max_y + 120),  # Slightly taller for better proportions
            margin=dict(l=30, r=30, t=80, b=30),  # More balanced margins
            dragmode="pan",
            hovermode="closest",
            showlegend=False,
            # Add subtle grid for better visual structure
            annotations=[
                dict(
                    text="<i style='color: rgba(255,255,255,0.4); font-size: 11px;'>💡 Click on column names to explore connections</i>",
                    showarrow=False,
                    x=0.5,
                    y=-0.05,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=11, color="rgba(255,255,255,0.6)")
                )
            ]
        )
        
        return fig

# Configure page settings
st.set_page_config(
    page_title="Penguin Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified configuration without color customization

# Logo configuration
LOGO_PATH = os.getenv("APP_LOGO_PATH", "")

# Simplified CSS without color customization
def load_custom_css():
    """Load minimalist CSS with company branding and dark theme."""
    st.markdown("""
    <style>
    /* Import clean, professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Root variables for company colors */
    :root {
        --primary-dark: #071D49;
        --accent-blue: #5BC2E7;
        --secondary-dark: #10182C;
        --text-light: #ffffff;
        --text-muted: #e0e0e0;
        --background-dark: #0a0a0a;
    }
    
    /* Global dark theme */
    .stApp {
        background-color: var(--background-dark);
        color: var(--text-light);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: transparent;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary-dark) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(91, 194, 231, 0.1);
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .logo-container img {
        max-height: 60px;
        width: auto;
        filter: brightness(0) invert(1);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--text-light);
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--primary-dark);
        border-right: 1px solid rgba(91, 194, 231, 0.2);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--text-light);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #4A9FD9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(91, 194, 231, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(91, 194, 231, 0.4);
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background-color: rgba(7, 29, 73, 0.3);
        border: 1px solid rgba(91, 194, 231, 0.2);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: rgba(16, 24, 44, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(91, 194, 231, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(7, 29, 73, 0.3);
        border: 1px solid rgba(91, 194, 231, 0.2);
        border-radius: 8px;
        color: var(--text-light);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: rgba(16, 24, 44, 0.7);
        color: var(--text-light);
        border: 1px solid rgba(91, 194, 231, 0.3);
        border-radius: 6px;
    }
    
    /* File uploader styling */
    .css-1cpxqw2 {
        background-color: rgba(7, 29, 73, 0.3);
        border: 2px dashed rgba(91, 194, 231, 0.5);
        border-radius: 8px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--accent-blue);
    }
    
    /* Hide footer and header but keep MainMenu for sidebar toggle */
    footer {visibility: hidden;}
    
    /* Ensure sidebar toggle is always visible and positioned correctly */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
        position: fixed !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
        z-index: 999999 !important;
        background-color: var(--accent-blue) !important;
        border-radius: 4px !important;
        padding: 0.25rem !important;
    }
    
    /* Style the sidebar toggle button */
    [data-testid="collapsedControl"] svg {
        color: white !important;
        width: 1rem !important;
        height: 1rem !important;
    }
    
    /* Processing Status CLI Container */
    .cli-container {
        background-color: rgba(16, 24, 44, 0.8);
        border: 1px solid rgba(91, 194, 231, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.4;
        color: var(--text-light);
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .processing-header {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary-dark) 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(91, 194, 231, 0.2);
    }
    
    .processing-step {
        background-color: rgba(7, 29, 73, 0.3);
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 3px solid var(--accent-blue);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4A9FD9;
    }
    </style>
    """, unsafe_allow_html=True)

class StringMatcherApp:
    """Main application class for Penguin Dashboard."""
    
    def __init__(self):
        self.base_dir = Path("d:\\String Matcher\\Main")
        self.temp_dir = None
        self.processing_status = {
            'is_processing': False,
            'current_step': '',
            'progress': 0,
            'logs': []
        }
        
        # Initialize session state
        if 'uploaded_file_processed' not in st.session_state:
            st.session_state.uploaded_file_processed = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
    
    def render_header(self):
        """Render the minimalist application header with company branding."""
        # Check if logo exists
        logo_path = "image.png"
        logo_exists = os.path.exists(logo_path)
        
        if logo_exists:
            # Encode logo as base64 for embedding
            with open(logo_path, "rb") as img_file:
                logo_b64 = base64.b64encode(img_file.read()).decode()
            
            logo_html = f'<div class="logo-container"><img src="data:image/png;base64,{logo_b64}" alt="Company Logo"></div>'
        else:
            logo_html = ''
        
        st.markdown(f"""
        <div class="header-container">
            {logo_html}
            <div style="text-align: center;">
                <h1 style="color: var(--text-light); margin-bottom: 0.5rem; font-weight: 600;">Penguin</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the minimalist sidebar with essential controls."""
        with st.sidebar:
            # File upload section
            upload_mode = st.radio(
                "Mode",
                ["New Analysis", "Load Results"]
            )
            
            if upload_mode == "New Analysis":
                uploaded_file = st.file_uploader(
                    "Upload File",
                    type=['xlsx', 'csv'],
                    key="new_analysis_file"
                )
                results_file = None
            else:
                uploaded_file = None
                results_file = st.file_uploader(
                    "Load Results",
                    type=['csv'],
                    key="existing_results_file"
                )
            
            st.markdown("---")
            
            # Matching configuration - only show for New Analysis
            if upload_mode == "New Analysis":
                semantic_weight = st.slider(
                    "Semantic Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1
                )
                
                fuzzy_weight = 1.0 - semantic_weight
                
                # Threshold settings
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=50,
                    max_value=100,
                    value=75
                )
                
                record_threshold = st.slider(
                    "Record Threshold",
                    min_value=50,
                    max_value=100,
                    value=80
                )
            else:
                # Default values when loading results
                semantic_weight = 0.6
                fuzzy_weight = 0.4
                similarity_threshold = 75
                record_threshold = 80
            
            st.markdown("---")
            
            # Results Configuration
            st.markdown("**Results Configuration**")
            
            high_threshold = st.slider(
                "High Confidence Threshold",
                min_value=60,
                max_value=95,
                value=90,
                help="Minimum score for high confidence category"
            )
            
            medium_threshold = st.slider(
                "Medium Confidence Threshold",
                min_value=30,
                max_value=80,
                value=70,
                help="Minimum score for medium confidence category"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                high_color = st.color_picker(
                    "High",
                    value="#2E8B57",
                    help="Color for high confidence matches"
                )
            with col2:
                medium_color = st.color_picker(
                    "Med",
                    value="#FF8C00",
                    help="Color for medium confidence matches"
                )
            with col3:
                low_color = st.color_picker(
                    "Low",
                    value="#DC143C",
                    help="Color for low confidence matches"
                )
            
            st.markdown("---")
            
            # Process button
            if upload_mode == "New Analysis":
                process_button = st.button(
                    "Start Analysis",
                    disabled=uploaded_file is None or self.processing_status['is_processing'],
                    use_container_width=True
                )
                load_results_button = False
            else:
                process_button = False
                load_results_button = st.button(
                    "Load Results",
                    disabled=results_file is None,
                    use_container_width=True
                )
            
            return {
                'uploaded_file': uploaded_file,
                'results_file': results_file,
                'upload_mode': upload_mode,
                'semantic_weight': semantic_weight,
                'fuzzy_weight': fuzzy_weight,
                'similarity_threshold': similarity_threshold,
                'record_threshold': record_threshold,
                'process_button': process_button,
                'load_results_button': load_results_button,
                'high_threshold': high_threshold,
                'medium_threshold': medium_threshold,
                'high_color': high_color,
                'medium_color': medium_color,
                'low_color': low_color
            }
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        file_path = os.path.join(self.temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for loaded results."""
        try:
            total_matches = len(df)
            
            # Calculate confidence category counts
            confidence_counts = df['confidence_category'].value_counts().to_dict()
            
            # Calculate percentages
            high_count = confidence_counts.get('high', 0)
            medium_count = confidence_counts.get('medium', 0)
            low_count = confidence_counts.get('low', 0)
            
            high_pct = (high_count / total_matches * 100) if total_matches > 0 else 0
            medium_pct = (medium_count / total_matches * 100) if total_matches > 0 else 0
            low_pct = (low_count / total_matches * 100) if total_matches > 0 else 0
            
            # Calculate average confidence score
            avg_confidence = df['unified_confidence_score'].mean() if 'unified_confidence_score' in df.columns else 0
            
            return {
                'total_matches': total_matches,
                'high_matches': high_count,
                'medium_matches': medium_count,
                'low_matches': low_count,
                'high_percentage': high_pct,
                'medium_percentage': medium_pct,
                'low_percentage': low_pct,
                'average_confidence': avg_confidence
            }
        except Exception as e:
            st.error(f"Error calculating summary stats: {str(e)}")
            return {
                'total_matches': 0,
                'high_matches': 0,
                'medium_matches': 0,
                'low_matches': 0,
                'high_percentage': 0,
                'medium_percentage': 0,
                'low_percentage': 0,
                'average_confidence': 0
            }
    
    def calculate_missing_columns(self, df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
        """Calculate missing unified_confidence_score and confidence_category columns."""
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Calculate unified_confidence_score if missing
            if 'unified_confidence_score' not in result_df.columns:
                # Try to find available similarity columns
                semantic_col = None
                fuzzy_col = None
                
                # Look for semantic similarity columns
                for col in result_df.columns:
                    if 'semantic' in col.lower() and 'similarity' in col.lower():
                        semantic_col = col
                        break
                
                # Look for fuzzy similarity columns
                for col in result_df.columns:
                    if 'fuzzy' in col.lower() and 'similarity' in col.lower():
                        fuzzy_col = col
                        break
                
                # If we have both semantic and fuzzy, calculate weighted average
                if semantic_col and fuzzy_col:
                    semantic_weight = settings.get('semantic_weight', 0.6)
                    fuzzy_weight = settings.get('fuzzy_weight', 0.4)
                    
                    result_df['unified_confidence_score'] = (
                        result_df[semantic_col] * semantic_weight + 
                        result_df[fuzzy_col] * fuzzy_weight
                    )
                elif semantic_col:
                    # Use semantic similarity only
                    result_df['unified_confidence_score'] = result_df[semantic_col]
                elif fuzzy_col:
                    # Use fuzzy similarity only
                    result_df['unified_confidence_score'] = result_df[fuzzy_col]
                else:
                    # Try to find any similarity column
                    similarity_cols = [col for col in result_df.columns if 'similarity' in col.lower()]
                    if similarity_cols:
                        result_df['unified_confidence_score'] = result_df[similarity_cols[0]]
                    else:
                        # Default to 50% if no similarity columns found
                        result_df['unified_confidence_score'] = 50.0
                        st.warning("⚠️ No similarity columns found. Using default confidence score of 50%.")
            
            # Calculate confidence_category if missing
            if 'confidence_category' not in result_df.columns:
                def categorize_confidence(score):
                    if score >= 90:
                        return 'High'
                    elif score >= 70:
                        return 'Medium'
                    else:
                        return 'Low'
                
                result_df['confidence_category'] = result_df['unified_confidence_score'].apply(categorize_confidence)
            
            # Confidence color column removed - no longer using color coding
            
            return result_df
            
        except Exception as e:
            st.error(f"Error calculating missing columns: {str(e)}")
            return df
    
    def run_comprehensive_workflow(self, input_file: str, settings: Dict) -> bool:
        """Execute the comprehensive workflow with real-time status updates."""
        try:
            # Use the input file directly instead of copying to avoid network access issues
            input_path = Path(input_file)
            
            # Import and run workflow directly
            sys.path.insert(0, str(self.base_dir))
            
            # Update processing logs
            st.session_state.processing_logs.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': 'Starting comprehensive workflow...'
            })
            
            # Execute cdlist_v2.py directly with full path to avoid file not found errors
            cmd = [
                sys.executable, "cdlist_v2.py", str(input_path),
                "--cross-header-match",
                "--record-output", "test_record_match.csv",
                "--output", "hybrid_linkages.csv",
                "--threshold", str(settings['similarity_threshold']),
                "--record-threshold", str(settings['record_threshold'])
            ]
            
            st.session_state.processing_logs.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': f'Executing: {" ".join(cmd)}'
            })
            
            # Create process with real-time output capture
            process = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor process output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    st.session_state.processing_logs.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': output.strip()
                    })
            
            return_code = process.poll()
            
            if return_code == 0:
                st.session_state.processing_logs.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': 'Step 1 completed: Column analysis finished'
                })
                
                # Execute transform_data.py
                cmd2 = [sys.executable, "transform_data.py"]
                
                st.session_state.processing_logs.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': 'Starting data transformation...'
                })
                
                process2 = subprocess.Popen(
                    cmd2,
                    cwd=str(self.base_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                while True:
                    output = process2.stdout.readline()
                    if output == '' and process2.poll() is not None:
                        break
                    if output:
                        st.session_state.processing_logs.append({
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'message': output.strip()
                        })
                
                return_code2 = process2.poll()
                
                if return_code2 == 0:
                    st.session_state.processing_logs.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': '✅ Analysis completed successfully!'
                    })
                    return True
                else:
                    st.session_state.processing_logs.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': '❌ Data transformation failed'
                    })
                    return False
            else:
                st.session_state.processing_logs.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': '❌ Column analysis failed'
                })
                return False
            
        except Exception as e:
            st.session_state.processing_logs.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': f"ERROR: {str(e)}"
            })
            return False
    
    def load_analysis_results(self) -> Optional[pd.DataFrame]:
        """Load the cell_expansion results."""
        results_file = self.base_dir / "enhanced_hybrid_linkages.csv"
        
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                return df
            except Exception as e:
                st.error(f"Error loading results: {str(e)}")
                return None
        else:
            st.warning("Results file not found. Please run analysis first.")
            return None
    
    def render_cli_interface(self):
        """Render cell_expansion real-time CLI interface during processing."""
        if self.processing_status['is_processing'] or st.session_state.processing_logs:
            # Enhanced header with status indicator
            status_icon = "🔄" if self.processing_status['is_processing'] else "✅"
            st.markdown(
                f'''
                <div class="processing-header">
                    <h3 style="margin: 0; color: var(--text-light); display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2em;">{status_icon}</span>
                        Processing Status
                    </h3>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
            # Create organized container
            with st.container():
                # Current step and progress
                if self.processing_status['is_processing']:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(
                            f'''
                            <div class="processing-step">
                                <strong>Current Step:</strong> {self.processing_status['current_step']}
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        progress_percentage = int(self.processing_status['progress'] * 100)
                        st.metric("Progress", f"{progress_percentage}%")
                    
                    # Progress bar
                    st.progress(self.processing_status['progress'])
                    st.markdown("---")
                
                # Enhanced logs display
                if st.session_state.processing_logs:
                    # Log summary
                    total_logs = len(st.session_state.processing_logs)
                    recent_logs = st.session_state.processing_logs[-20:]  # Show last 20 lines
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown("**📋 Processing Logs**")
                    with col2:
                        st.markdown(f"**Total:** {total_logs}")
                    with col3:
                        st.markdown(f"**Showing:** {len(recent_logs)}")
                    
                    # Formatted log display
                    log_entries = []
                    for log in recent_logs:
                        timestamp = log['timestamp']
                        message = log['message']
                        
                        # Add color coding for different message types
                        if '✅' in message or 'completed successfully' in message.lower():
                            color = '#4CAF50'  # Green for success
                        elif '❌' in message or 'failed' in message.lower() or 'error' in message.lower():
                            color = '#F44336'  # Red for errors
                        elif 'starting' in message.lower() or 'executing' in message.lower():
                            color = '#2196F3'  # Blue for actions
                        else:
                            color = 'var(--text-light)'  # Default
                        
                        log_entries.append(
                            f'<span style="color: var(--accent-blue);">[{timestamp}]</span> '
                            f'<span style="color: {color};">{message}</span>'
                        )
                    
                    log_html = '\n'.join(log_entries)
                    
                    st.markdown(
                        f'<div class="cli-container">{log_html}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Auto-scroll indicator
                    if self.processing_status['is_processing']:
                        st.markdown(
                            '<p style="text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem;">'
                            '🔄 Live updates • Scroll up to see earlier logs</p>',
                            unsafe_allow_html=True
                        )
    
    def create_enhanced_visualizations(self, df: pd.DataFrame, viz_config: Dict) -> Tuple[go.Figure, go.Figure]:
        """Create enhanced bar chart visualizations with unified confidence scoring."""
        # Initialize calculator and visualization classes
        calculator = UnifiedScoreCalculator(viz_config['weights'])
        visualizer = EnhancedVisualization()
        
        # Calculate unified confidence scores
        df_enhanced = df.copy()
        df_enhanced['unified_confidence'] = df_enhanced.apply(calculator.calculate_unified_score, axis=1)
        
        # Apply basic categorization
        df_categorized = visualizer.apply_basic_categorization(df_enhanced, 'unified_confidence')
        
        # Create visualizations
        distribution_chart = visualizer.create_confidence_distribution_chart(df_categorized)
        
        return distribution_chart, df_categorized
    

    
    def render_visualization_config(self, sidebar_settings: Dict = None) -> Dict:
        """Return configuration for enhanced visualizations using sidebar settings."""
        if sidebar_settings:
            return {
                'weights': {'column': 0.33, 'semantic': 0.33, 'example': 0.34},
                'results_config': {
                    'high_threshold': sidebar_settings.get('high_threshold', 90),
                    'medium_threshold': sidebar_settings.get('medium_threshold', 70),
                    'high_color': sidebar_settings.get('high_color', '#2E8B57'),
                    'medium_color': sidebar_settings.get('medium_color', '#FF8C00'),
                    'low_color': sidebar_settings.get('low_color', '#DC143C')
                }
            }
        else:
            # Return default configuration if no sidebar settings provided
            return {
                'weights': {'column': 0.33, 'semantic': 0.33, 'example': 0.34},
                'results_config': {
                    'high_threshold': 90,
                    'medium_threshold': 70,
                    'high_color': '#2E8B57',
                    'medium_color': '#FF8C00',
                    'low_color': '#DC143C'
                }
            }


     
    def render_results_section(self, df: pd.DataFrame, sidebar_settings: Dict = None):
        """Render the enhanced results analysis section."""
        st.markdown("### Cell Expansion Analysis Results")
        
        # Get visualization configuration
        viz_config = self.render_visualization_config(sidebar_settings)
        
        # Create enhanced visualizations
        distribution_chart, df_categorized = self.create_enhanced_visualizations(df, viz_config)
        
        # Summary metrics with unified confidence
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Linkages",
                len(df_categorized),
                help="Total number of column linkages identified"
            )
        
        with col2:
            high_count = len(df_categorized[df_categorized['confidence_category'] == 'high'])
            st.metric(
                "High",
                high_count,
                help="Linkages with unified confidence ≥ 90%"
            )
        
        with col3:
            medium_count = len(df_categorized[df_categorized['confidence_category'] == 'medium'])
            st.metric(
                "Medium",
                medium_count,
                help="Linkages with unified confidence 70-89%"
            )
        
        with col4:
            low_count = len(df_categorized[df_categorized['confidence_category'] == 'low'])
            st.metric(
                "Low",
                low_count,
                help="Linkages with unified confidence < 70%"
            )
        
        with col5:
            avg_unified = df_categorized['unified_confidence'].mean()
            st.metric(
                "Avg Unified Score",
                f"{avg_unified:.1f}%",
                help="Average unified confidence score"
            )
        
        # Enhanced Visualizations with Chart Selection
        st.markdown("#### Score Analysis")
        
        # Chart selection dropdown
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Unified Confidence Distribution", "Semantic Similarity Distribution", "Column Name Similarity Distribution", "ER Diagram"],
            help="Choose which analysis chart to display"
        )
        
        # Display selected chart with interactive functionality
        viz = EnhancedVisualization()
        if chart_type == "Unified Confidence Distribution":
            chart_fig = distribution_chart
            score_column = 'unified_confidence'
            # Create range mapping that accounts for dynamic filtering
            all_ranges = [(i, i+9) for i in range(0, 90, 10)] + [(90, 100)]
            range_labels = [f"{i}-{i+9}" for i in range(0, 90, 10)] + ["90-100"]
            
            # Get the actual ranges that have data (matching the chart's filtered ranges)
            bins = list(range(0, 101, 10))
            df_temp = df_categorized.copy()
            df_temp['confidence_range'] = pd.cut(df_temp['unified_confidence'], bins=bins, labels=range_labels)
            range_counts = df_temp['confidence_range'].value_counts().sort_index()
            filtered_ranges = range_counts[range_counts > 0]
            
            # Create mapping from chart index to actual range
            ranges = [all_ranges[range_labels.index(label)] for label in filtered_ranges.index]
            
        elif chart_type == "Semantic Similarity Distribution":
            chart_fig = viz.create_semantic_similarity_chart(df_categorized)
            score_column = 'semantic_similarity'
            # Create range mapping that accounts for dynamic filtering
            all_ranges = [(-100, -80), (-80, -60), (-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
            range_labels = ['-100 to -80', '-80 to -60', '-60 to -40', '-40 to -20', '-20 to 0',
                           '0 to 20', '20 to 40', '40 to 60', '60 to 80', '80 to 100']
            
            # Get the actual ranges that have data (matching the chart's filtered ranges)
            bins = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
            df_temp = df_categorized.copy()
            df_temp['semantic_range'] = pd.cut(df_temp['semantic_similarity'], bins=bins, labels=range_labels)
            range_counts = df_temp['semantic_range'].value_counts().sort_index()
            filtered_ranges = range_counts[range_counts > 0]
            
            # Create mapping from chart index to actual range
            ranges = [all_ranges[range_labels.index(label)] for label in filtered_ranges.index]
        elif chart_type == "Column Name Similarity Distribution":
            chart_fig = viz.create_column_similarity_chart(df_categorized)
            score_column = 'column_name_similarity'
            # Create range mapping that accounts for dynamic filtering
            all_ranges = [(i, i+9) for i in range(0, 90, 10)] + [(90, 100)]
            range_labels = [f"{i}-{i+9}" for i in range(0, 90, 10)] + ["90-100"]
            
            # Get the actual ranges that have data (matching the chart's filtered ranges)
            bins = list(range(0, 101, 10))
            df_temp = df_categorized.copy()
            df_temp['similarity_range'] = pd.cut(df_temp['column_name_similarity'], bins=bins, labels=range_labels)
            range_counts = df_temp['similarity_range'].value_counts().sort_index()
            filtered_ranges = range_counts[range_counts > 0]
            
            # Create mapping from chart index to actual range
            ranges = [all_ranges[range_labels.index(label)] for label in filtered_ranges.index]
        elif chart_type == "ER Diagram":
            chart_fig = viz.create_er_diagram(df_categorized)
            score_column = None
            ranges = []
        
        # Display the chart
        selected_points = st.plotly_chart(chart_fig, use_container_width=True, on_select="rerun")
        
        # Handle drill-down functionality
        if chart_type == "ER Diagram":
            self.render_er_diagram_interactions(df_categorized, selected_points)
        else:
            self.render_drill_down_section(df_categorized, chart_type, score_column, ranges, selected_points)
        
        # Add exceptional matches section
        self.render_exceptional_matches_section(df_categorized)
        


        

        
        # Comprehensive Results Display with Color-Coded Confidence Categories
        self.render_comprehensive_results_display(df_categorized, viz_config)

    
    def render_comprehensive_results_display(self, df: pd.DataFrame, config: Dict):
        """Render comprehensive results display with configurable color-coded confidence categories."""
        st.markdown("#### 📊 Comprehensive Results Analysis")
        st.markdown("*All analyzed results with configurable color-coded confidence levels (configured in sidebar)*")
        
        # Get configuration from sidebar
        results_config = config.get('results_config', {})
        high_threshold = results_config.get('high_threshold', 70)
        medium_threshold = results_config.get('medium_threshold', 40)
        high_color = results_config.get('high_color', '#2E8B57')
        medium_color = results_config.get('medium_color', '#FF8C00')
        low_color = results_config.get('low_color', '#DC143C')
        
        # Recategorize confidence based on configurable thresholds
        def get_dynamic_confidence_category(score: float) -> str:
            if score >= high_threshold:
                return 'high'
            elif score >= medium_threshold:
                return 'medium'
            else:
                return 'low'
        
        # Apply dynamic categorization
        df_recategorized = df.copy()
        df_recategorized['confidence_category'] = df_recategorized['unified_confidence'].apply(get_dynamic_confidence_category)
        
        # Create color mapping
        color_map = {
            'high': high_color,
            'medium': medium_color,
            'low': low_color
        }
        
        # Display threshold information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**High Confidence** (≥{high_threshold}%): {df_recategorized[df_recategorized['confidence_category'] == 'high'].shape[0]} records")
        with col2:
            st.markdown(f"**Medium Confidence** ({medium_threshold}-{high_threshold-1}%): {df_recategorized[df_recategorized['confidence_category'] == 'medium'].shape[0]} records")
        with col3:
            st.markdown(f"**Low Confidence** (<{medium_threshold}%): {df_recategorized[df_recategorized['confidence_category'] == 'low'].shape[0]} records")
        
        # Use all recategorized data without filtering
        filtered_df = df_recategorized
        display_df = filtered_df.copy()
        
        # Format display dataframe with color coding
        display_df_formatted = display_df.copy()
        
        # Round numerical columns for better readability
        numerical_cols = ['semantic_similarity', 'column_name_similarity', 'unified_confidence', 'cell_match_score']
        for col in numerical_cols:
            if col in display_df_formatted.columns:
                display_df_formatted[col] = display_df_formatted[col].round(2)
        
        # Create styled dataframe with color coding
        def style_confidence_category(row):
            category = row['confidence_category']
            color = color_map.get(category, '#FFFFFF')
            # Convert hex to rgba for better readability
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"
            return [f'background-color: {rgba_color}'] * len(row)
        
        # Apply styling
        styled_df = display_df_formatted.style.apply(style_confidence_category, axis=1)
        
        # Reorder columns for better presentation
        column_order = [
            'source_sheet', 'source_column', 'target_sheet', 'target_column',
            'unified_confidence', 'confidence_category', 'semantic_similarity',
            'column_name_similarity', 'cell_match_score'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in display_df_formatted.columns]
        other_columns = [col for col in display_df_formatted.columns if col not in available_columns]
        final_columns = available_columns + other_columns
        
        # Display the styled dataframe - fix Styler subscriptability issue
        # Reorder the original dataframe columns, then apply styling
        display_df_reordered = display_df_formatted[final_columns]
        styled_df_final = display_df_reordered.style.apply(style_confidence_category, axis=1)
        
        st.dataframe(
            styled_df_final,
            use_container_width=True,
            height=400
        )
        
        # Download button
        if len(filtered_df) > 0:
            download_csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=download_csv,
                file_name=f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_comprehensive_results"
            )
    
    def render_exceptional_matches_section(self, df: pd.DataFrame):
        """Render exceptional matches section - matches with low unified scores but perfect cell value matches."""
        st.markdown("#### Exceptional Matches")
        st.markdown("*Matches with low semantic similarity but perfect cell value alignment*")
        
        # Calculate unified confidence scores on the fly
        calculator = UnifiedScoreCalculator()
        df_with_unified = df.copy()
        df_with_unified['unified_confidence'] = df_with_unified.apply(calculator.calculate_unified_score, axis=1)
        
        # Filter for exceptional matches: low semantic similarity but perfect cell match score
        # Exceptional = semantic_similarity < 20 AND cell_match_score == 100
        # This identifies cases where column names are semantically different but data values are identical
        exceptional_condition = (
            (df_with_unified['semantic_similarity'] < 60.0) & 
            (df_with_unified['cell_match_score'] == 100.0)
        )
        exceptional_matches = df_with_unified[exceptional_condition]
        
        if len(exceptional_matches) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Exceptional Matches",
                    len(exceptional_matches),
                    help="Matches with semantic similarity < 20% but perfect cell value match (100%)"
                )
            
            with col2:
                avg_semantic = exceptional_matches['semantic_similarity'].mean()
                st.metric(
                    "Avg Semantic Score",
                    f"{avg_semantic:.1f}%",
                    help="Average semantic similarity of exceptional matches"
                )
            
            with col3:
                perfect_cell_matches = len(exceptional_matches[exceptional_matches['cell_match_score'] == 100.0])
                st.metric(
                    "Perfect Cell Matches",
                    perfect_cell_matches,
                    help="Number of matches with 100% cell value alignment"
                )
            
            # Display exceptional matches
            st.markdown("##### Exceptional Match Details")
            
            # Sort by cell_match_score (descending) then by unified_confidence (ascending)
            sorted_matches = exceptional_matches.sort_values(
                ['cell_match_score', 'unified_confidence'], 
                ascending=[False, True]
            )
            
            # Format for display - include unified_confidence column
            display_cols = ['source_sheet', 'source_column', 'target_sheet', 'target_column', 
                          'source_cell_value', 'target_cell_value', 'unified_confidence', 
                          'cell_match_score', 'column_name_similarity', 'semantic_similarity']
            
            available_cols = [col for col in display_cols if col in sorted_matches.columns]
            display_df = sorted_matches[available_cols].copy()
            
            # Round numerical columns
            for col in ['unified_confidence', 'column_name_similarity', 'semantic_similarity', 'cell_match_score']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
            
            # Add explanation
            st.info(
                "💡 **Why these are exceptional**: These matches have low semantic similarity scores & column similarrity score "
                "but show perfect alignment in actual cell values, suggesting they might be valid "
                "connections despite poor semantic understanding between column names."
            )
            
        else:
            st.info("No exceptional matches found (semantic similarity < 20% with perfect cell value match).")
    
    def render_drill_down_section(self, df: pd.DataFrame, chart_type: str, score_column: str, ranges: list, selected_points):
        """Render drill-down section showing filtered records based on chart selection."""
        # Skip drill-down for ER diagram as it has different interaction model
        if chart_type == "ER Diagram":
            return
            
        if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
            # Get the selected bar index
            try:
                selected_indices = selected_points.selection.get('points', [])
                if selected_indices:
                    # Get the first selected point
                    point_index = selected_indices[0].get('point_index', 0)
                    
                    # Get the corresponding range
                    if point_index < len(ranges):
                        min_val, max_val = ranges[point_index]
                        
                        # Filter data based on the selected range
                        if chart_type == "Semantic Similarity Distribution":
                            filtered_df = df[(df[score_column] >= min_val) & (df[score_column] <= max_val)]
                            range_label = f"{min_val} to {max_val}"
                        else:
                            filtered_df = df[(df[score_column] >= min_val) & (df[score_column] <= max_val)]
                            range_label = f"{min_val}% to {max_val}%"
                        
                        # Display drill-down results
                        st.markdown(f"##### Drill-Down: {chart_type.replace(' Distribution', '')} Range {range_label}")
                        
                        if len(filtered_df) > 0:
                            # Summary metrics for the filtered data
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Records in Range",
                                    len(filtered_df),
                                    help=f"Number of linkages in the {range_label} range"
                                )
                            
                            with col2:
                                avg_score = filtered_df[score_column].mean()
                                st.metric(
                                    f"Average {score_column.replace('_', ' ').title()}",
                                    f"{avg_score:.1f}{'%' if 'confidence' in score_column or 'similarity' in score_column else ''}",
                                    help=f"Average score in this range"
                                )
                            
                            with col3:
                                avg_unified = filtered_df['unified_confidence'].mean()
                                st.metric(
                                    "Average Unified Score",
                                    f"{avg_unified:.1f}%",
                                    help="Average unified confidence for this range"
                                )
                            
                            # Display filtered records
                            display_cols = ['source_sheet', 'source_column', 'target_sheet', 'target_column', 
                                          'source_cell_value', 'target_cell_value', 'unified_confidence', 
                                          'semantic_similarity', 'column_name_similarity']
                            
                            available_cols = [col for col in display_cols if col in filtered_df.columns]
                            display_df = filtered_df[available_cols].copy()
                            
                            # Round numerical columns
                            for col in ['unified_confidence', 'semantic_similarity', 'column_name_similarity']:
                                if col in display_df.columns:
                                    display_df[col] = display_df[col].round(2)
                            
                            # Sort by the selected score column
                            display_df = display_df.sort_values(by=score_column, ascending=False)
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                height=min(400, len(display_df) * 35 + 50)
                            )
                            
                            # Download option for filtered data
                            if len(filtered_df) > 10:
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download {len(filtered_df)} Records",
                                    data=csv,
                                    file_name=f"drill_down_{chart_type.lower().replace(' ', '_')}_{range_label.replace(' ', '_').replace('%', 'pct')}.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.info(f"No records found in the {range_label} range.")
            except Exception as e:
                st.error(f"Error processing selection: {str(e)}")
        else:
            st.info("💡 **Tip**: Click on any bar in the chart above to see detailed records for that range!")
    
    def render_er_diagram_interactions(self, df: pd.DataFrame, selected_points):
        """Handle ER diagram column click interactions and display related linkages with optimized performance."""
        # Use cached unified confidence scores if available
        if not hasattr(self, '_cached_df_with_confidence') or len(self._cached_df_with_confidence) != len(df):
            df_with_confidence = df.copy()
            if 'unified_confidence' not in df_with_confidence.columns:
                calculator = UnifiedScoreCalculator()
                df_with_confidence['unified_confidence'] = df_with_confidence.apply(calculator.calculate_unified_score, axis=1)
            self._cached_df_with_confidence = df_with_confidence
        else:
            df_with_confidence = self._cached_df_with_confidence
        
        if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
            try:
                selected_indices = selected_points.selection.get('points', [])
                if selected_indices:
                    # Get the selected point's custom data (column identifier)
                    point_data = selected_indices[0]
                    if 'customdata' in point_data and point_data['customdata']:
                        selected_column = point_data['customdata'][0]
                        
                        # Parse table and column name
                        if '.' in selected_column:
                            table_name, column_name = selected_column.split('.', 1)
                            
                            # Optimized filtering using boolean indexing with pre-computed masks
                            source_mask = (df_with_confidence['source_sheet'] == table_name) & (df_with_confidence['source_column'] == column_name)
                            target_mask = (df_with_confidence['target_sheet'] == table_name) & (df_with_confidence['target_column'] == column_name)
                            related_linkages = df_with_confidence[source_mask | target_mask].copy()
                            
                            if len(related_linkages) > 0:
                                st.markdown(f"##### 🔗 Linkages for Column: **{selected_column}**")
                                
                                # Summary metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Total Connections",
                                        len(related_linkages),
                                        help=f"Number of linkages involving {selected_column}"
                                    )
                                
                                with col2:
                                    avg_confidence = related_linkages['unified_confidence'].mean()
                                    st.metric(
                                        "Average Confidence",
                                        f"{avg_confidence:.1f}%",
                                        help="Average unified confidence for these linkages"
                                    )
                                
                                with col3:
                                    high_confidence_count = len(related_linkages[related_linkages['unified_confidence'] >= 70])
                                    st.metric(
                                        "High Confidence",
                                        high_confidence_count,
                                        help="Linkages with confidence ≥ 70%"
                                    )
                                
                                # Display related linkages table
                                display_cols = ['source_sheet', 'source_column', 'target_sheet', 'target_column', 
                                              'unified_confidence', 'semantic_similarity', 'column_name_similarity']
                                
                                available_cols = [col for col in display_cols if col in related_linkages.columns]
                                display_df = related_linkages[available_cols].copy()
                                
                                # Round numerical columns
                                for col in ['unified_confidence', 'semantic_similarity', 'column_name_similarity']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].round(2)
                                
                                # Sort by confidence
                                display_df = display_df.sort_values(by='unified_confidence', ascending=False)
                                
                                # Style the dataframe to highlight the selected column
                                def highlight_selected_column(row):
                                    styles = [''] * len(row)
                                    if (row['source_sheet'] == table_name and row['source_column'] == column_name) or \
                                       (row['target_sheet'] == table_name and row['target_column'] == column_name):
                                        styles = ['background-color: rgba(91, 194, 231, 0.2)'] * len(row)
                                    return styles
                                
                                styled_df = display_df.style.apply(highlight_selected_column, axis=1)
                                
                                st.dataframe(
                                    styled_df,
                                    use_container_width=True,
                                    height=min(400, len(display_df) * 35 + 50)
                                )
                                
                                # Download option
                                if len(related_linkages) > 0:
                                    csv = display_df.to_csv(index=False)
                                    st.download_button(
                                        label=f"📥 Download {len(related_linkages)} Linkages",
                                        data=csv,
                                        file_name=f"linkages_{table_name}_{column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info(f"No linkages found for column **{selected_column}**")
                        else:
                            st.warning("Invalid column selection format")
            except Exception as e:
                st.error(f"Error processing ER diagram selection: {str(e)}")
        else:
            st.info("💡 **Tip**: Click on any column name in the ER diagram above to see its related linkages!")
    
    def run(self):
        """Main application entry point."""
        # Load custom CSS
        load_custom_css()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Main content area
        if settings['process_button'] and settings['uploaded_file'] is not None:
            # Start processing
            self.processing_status['is_processing'] = True
            st.session_state.processing_logs = []
            
            with st.spinner("Processing uploaded file..."):
                # Save uploaded file
                input_file = self.save_uploaded_file(settings['uploaded_file'])
                
                # Update processing status
                self.processing_status['current_step'] = "Initializing analysis..."
                self.processing_status['progress'] = 0.1
                
                # Run workflow
                success = self.run_comprehensive_workflow(input_file, settings)
                
                if success:
                    # Load results
                    st.session_state.analysis_results = self.load_analysis_results()
                    st.session_state.uploaded_file_processed = True
                    
                    self.processing_status['is_processing'] = False
                    st.success("✅ Analysis completed successfully!")
                    st.rerun()
                else:
                    self.processing_status['is_processing'] = False
                    st.error("❌ Analysis failed. Please check the logs below.")
        
        elif settings['load_results_button'] and settings['results_file'] is not None:
            # Load existing results
            with st.spinner("Loading existing results..."):
                try:
                    # Read the CSV file
                    results_df = pd.read_csv(settings['results_file'])
                    
                    # Check if required columns exist, if not, calculate them
                    required_columns = ['unified_confidence_score', 'confidence_category']
                    missing_columns = [col for col in required_columns if col not in results_df.columns]
                    
                    if missing_columns:
                        st.info(f"Calculating missing columns: {', '.join(missing_columns)}")
                        results_df = self.calculate_missing_columns(results_df, settings)
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'detailed_results': results_df,
                        'summary_stats': self.calculate_summary_stats(results_df)
                    }
                    st.session_state.uploaded_file_processed = True
                    st.success("✅ Results loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading results file: {str(e)}")
        
        # Render CLI interface if processing or logs exist
        self.render_cli_interface()
        
        # Display results if available
        if st.session_state.analysis_results is not None:
            # Extract the DataFrame from the results dict
            if isinstance(st.session_state.analysis_results, dict) and 'detailed_results' in st.session_state.analysis_results:
                self.render_results_section(st.session_state.analysis_results['detailed_results'], settings)
            else:
                # Handle case where analysis_results is already a DataFrame
                self.render_results_section(st.session_state.analysis_results, settings)
        elif not self.processing_status['is_processing']:
            # Show welcome message
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; color: var(--text-muted);">
                <h3 style="color: var(--text-light); margin-bottom: 1rem;">Welcome</h3>
                <p style="font-size: 1.1rem; margin-bottom: 2rem;">Upload your data file to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)

# Application entry point
if __name__ == "__main__":
    app = StringMatcherApp()
    app.run()
