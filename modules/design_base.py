"""
Backend module providing the base designer class.
"""

import re
import pandas as pd


# ###########################
# Base class for designers
# ###########################
class DesignerBaseClass:
    """
    Base class for designers (same methods for all approaches).
    """

    def export_to_excel(self, buffer) -> None:
        """Export design to excel.
        """
        df = self._get_design_dataframe()
        meta = self._get_metadata()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Design")
            md = pd.DataFrame(list(meta.items()), columns=["key", "value"])
            md.to_excel(writer, index=False, sheet_name="metadata")