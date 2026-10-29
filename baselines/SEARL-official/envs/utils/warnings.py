# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Suppress MCP validation warnings through patches installed before MCP imports.'

import logging
import warnings
from typing import Any

def suppress_mcp_warnings():
    'Suppress MCP warnings.'

    warnings.filterwarnings("ignore")

    loggers_to_suppress = [
        'mcp',
        'mcp.client',
        'mcp.client.sse',
        'mcp.client.stdio',
        'mcp.client.streamable_http',
        'mcp.shared',
        'mcp.shared.session',
        'pydantic',
        'httpx',
        'httpcore',
        'qwen_agent',
    ]

    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

    original_warning = logging.warning
    original_logger_warning = logging.Logger.warning

    def patched_warning(msg, *args, **kwargs):

        msg_str = str(msg)
        if any(keyword in msg_str.lower() for keyword in [
            'failed to validate notification',
            'validation errors',
            'pydantic',
            'servernotification',
            'cancellednotification',
            'progressnotification',
            'loggingmessagenotification',
            'resourceupdatednotification',
            'resourcelistchangednotification',
            'toollistchangednotification',
            'promptlistchangednotification',
            'literal_error',
            'input_value=\'ping\'',
            'field required'
        ]):
            return

        return original_warning(msg, *args, **kwargs)

    def patched_logger_warning(self, msg, *args, **kwargs):

        if hasattr(self, 'name') and self.name and any(name in self.name.lower() for name in [
            'mcp', 'pydantic', 'httpx', 'httpcore', 'qwen_agent'
        ]):
            return

        msg_str = str(msg)
        if any(keyword in msg_str.lower() for keyword in [
            'failed to validate notification',
            'validation errors',
            'pydantic',
            'servernotification',
            'cancellednotification',
            'progressnotification',
            'loggingmessagenotification',
            'resourceupdatednotification',
            'resourcelistchangednotification',
            'toollistchangednotification',
            'promptlistchangednotification',
            'literal_error',
            'input_value=\'ping\'',
            'field required'
        ]):
            return

        return original_logger_warning(self, msg, *args, **kwargs)

    logging.warning = patched_warning
    logging.Logger.warning = patched_logger_warning

    try:
        import mcp.shared.session

        if hasattr(mcp.shared.session, 'logging'):
            original_session_warning = mcp.shared.session.logging.warning

            def patched_session_warning(msg, *args, **kwargs):
                msg_str = str(msg)
                if 'failed to validate' in msg_str.lower() or 'validation errors' in msg_str.lower():
                    return
                return original_session_warning(msg, *args, **kwargs)

            mcp.shared.session.logging.warning = patched_session_warning
    except ImportError:
        pass

    print('MCP warning suppression enabled.')

def patch_ray_worker_logging():
    'Apply logging suppression inside Ray workers.'
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    suppress_mcp_warnings()

suppress_mcp_warnings()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
