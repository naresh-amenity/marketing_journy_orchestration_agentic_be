# Journey Reports

This directory stores the generated journey report documents created by the JourneyTool.

## Overview

The JourneyTool allows users to:
1. Create marketing journeys
2. Check journey status
3. Generate journey reports as Word documents (stored in this directory)
4. Update journey reports with persona files

## Usage

The generated Word documents follow a naming convention:
```
journey_[journey_id]_[timestamp].docx
```

For example:
```
journey_5msgtpa7gt_20240325_121530.docx
```

These documents contain the structured journey report data formatted into a readable Word document.

## Folder Structure

- Word documents (.docx) - Journey reports generated from API data
- Any uploaded persona files will be processed but not stored in this directory

## Notes

- Do not delete this directory as it is required for the JourneyTool to function properly
- Reports are automatically generated when requesting a journey report through the API 