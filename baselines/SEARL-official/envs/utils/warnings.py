# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\nMCP \u8b66\u544a\u5c4f\u853d\u5de5\u5177\n\n\u8fd9\u4e2a\u6a21\u5757\u901a\u8fc7\u7334\u5b50\u8865\u4e01\u7684\u65b9\u5f0f\u6765\u5c4f\u853d MCP \u670d\u52a1\u5668\u4ea7\u751f\u7684\u9a8c\u8bc1\u8b66\u544a\u3002\n\u5728\u5bfc\u5165 MCP \u76f8\u5173\u6a21\u5757\u4e4b\u524d\u5148\u5bfc\u5165\u8fd9\u4e2a\u6a21\u5757\u3002\n'

import logging
import warnings
from typing import Any

def suppress_mcp_warnings():
    '\u5c4f\u853d MCP \u76f8\u5173\u7684\u8b66\u544a\u4fe1\u606f'

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

    print('\u2705 MCP \u8b66\u544a\u5c4f\u853d\u5df2\u542f\u7528')

def patch_ray_worker_logging():
    '\u5728Ray worker\u4e2d\u4e5f\u5e94\u7528\u65e5\u5fd7\u5c4f\u853d'
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    suppress_mcp_warnings()

suppress_mcp_warnings()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
