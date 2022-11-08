# Diff Details

Date : 2022-11-08 13:26:34

Directory /Users/yunuszhang/code/demo/demo_study_vditor/vditor/src/ts/wysiwyg

Total : 96 files,  -8379 codes, -341 comments, -766 blanks, all -9486 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [src/ts/constants.ts](/src/ts/constants.ts) | TypeScript | -62 | -2 | -4 | -68 |
| [src/ts/devtools/index.ts](/src/ts/devtools/index.ts) | TypeScript | -78 | 0 | -6 | -84 |
| [src/ts/export/index.ts](/src/ts/export/index.ts) | TypeScript | -76 | 0 | -6 | -82 |
| [src/ts/hint/index.ts](/src/ts/hint/index.ts) | TypeScript | -346 | -6 | -21 | -373 |
| [src/ts/ir/expandMarker.ts](/src/ts/ir/expandMarker.ts) | TypeScript | -63 | -5 | -12 | -80 |
| [src/ts/ir/highlightToolbarIR.ts](/src/ts/ir/highlightToolbarIR.ts) | TypeScript | -83 | 0 | -15 | -98 |
| [src/ts/ir/index.ts](/src/ts/ir/index.ts) | TypeScript | -221 | -12 | -24 | -257 |
| [src/ts/ir/input.ts](/src/ts/ir/input.ts) | TypeScript | -192 | -23 | -26 | -241 |
| [src/ts/ir/process.ts](/src/ts/ir/process.ts) | TypeScript | -205 | -3 | -13 | -221 |
| [src/ts/ir/processKeydown.ts](/src/ts/ir/processKeydown.ts) | TypeScript | -196 | -20 | -23 | -239 |
| [src/ts/markdown/abcRender.ts](/src/ts/markdown/abcRender.ts) | TypeScript | -25 | 0 | -3 | -28 |
| [src/ts/markdown/adapterRender.ts](/src/ts/markdown/adapterRender.ts) | TypeScript | -32 | -1 | -1 | -34 |
| [src/ts/markdown/anchorRender.ts](/src/ts/markdown/anchorRender.ts) | TypeScript | -18 | 0 | -2 | -20 |
| [src/ts/markdown/chartRender.ts](/src/ts/markdown/chartRender.ts) | TypeScript | -34 | 0 | -4 | -38 |
| [src/ts/markdown/codeRender.ts](/src/ts/markdown/codeRender.ts) | TypeScript | -46 | -1 | -8 | -55 |
| [src/ts/markdown/flowchartRender.ts](/src/ts/markdown/flowchartRender.ts) | TypeScript | -23 | 0 | -3 | -26 |
| [src/ts/markdown/getHTML.ts](/src/ts/markdown/getHTML.ts) | TypeScript | -10 | 0 | -2 | -12 |
| [src/ts/markdown/getMarkdown.ts](/src/ts/markdown/getMarkdown.ts) | TypeScript | -11 | 0 | -2 | -13 |
| [src/ts/markdown/graphvizRender.ts](/src/ts/markdown/graphvizRender.ts) | TypeScript | -42 | 0 | -8 | -50 |
| [src/ts/markdown/highlightRender.ts](/src/ts/markdown/highlightRender.ts) | TypeScript | -75 | -1 | -11 | -87 |
| [src/ts/markdown/lazyLoadImageRender.ts](/src/ts/markdown/lazyLoadImageRender.ts) | TypeScript | -52 | 0 | -5 | -57 |
| [src/ts/markdown/mathRender.ts](/src/ts/markdown/mathRender.ts) | TypeScript | -135 | -1 | -9 | -145 |
| [src/ts/markdown/mediaRender.ts](/src/ts/markdown/mediaRender.ts) | TypeScript | -74 | 0 | -5 | -79 |
| [src/ts/markdown/mermaidRender.ts](/src/ts/markdown/mermaidRender.ts) | TypeScript | -47 | 0 | -3 | -50 |
| [src/ts/markdown/mindmapRender.ts](/src/ts/markdown/mindmapRender.ts) | TypeScript | -72 | 0 | -3 | -75 |
| [src/ts/markdown/outlineRender.ts](/src/ts/markdown/outlineRender.ts) | TypeScript | -109 | 0 | -2 | -111 |
| [src/ts/markdown/plantumlRender.ts](/src/ts/markdown/plantumlRender.ts) | TypeScript | -30 | 0 | -3 | -33 |
| [src/ts/markdown/previewRender.ts](/src/ts/markdown/previewRender.ts) | TypeScript | -145 | 0 | -7 | -152 |
| [src/ts/markdown/setLute.ts](/src/ts/markdown/setLute.ts) | TypeScript | -23 | -2 | -1 | -26 |
| [src/ts/markdown/speechRender.ts](/src/ts/markdown/speechRender.ts) | TypeScript | -94 | 0 | -8 | -102 |
| [src/ts/outline/index.ts](/src/ts/outline/index.ts) | TypeScript | -43 | 0 | -5 | -48 |
| [src/ts/preview/image.ts](/src/ts/preview/image.ts) | TypeScript | -46 | -2 | -3 | -51 |
| [src/ts/preview/index.ts](/src/ts/preview/index.ts) | TypeScript | -244 | -6 | -16 | -266 |
| [src/ts/resize/index.ts](/src/ts/resize/index.ts) | TypeScript | -47 | 0 | -11 | -58 |
| [src/ts/sv/index.ts](/src/ts/sv/index.ts) | TypeScript | -111 | -2 | -13 | -126 |
| [src/ts/sv/inputEvent.ts](/src/ts/sv/inputEvent.ts) | TypeScript | -179 | -19 | -11 | -209 |
| [src/ts/sv/process.ts](/src/ts/sv/process.ts) | TypeScript | -203 | -5 | -15 | -223 |
| [src/ts/sv/processKeydown.ts](/src/ts/sv/processKeydown.ts) | TypeScript | -172 | -17 | -8 | -197 |
| [src/ts/tip/index.ts](/src/ts/tip/index.ts) | TypeScript | -34 | -1 | -5 | -40 |
| [src/ts/toolbar/Both.ts](/src/ts/toolbar/Both.ts) | TypeScript | -27 | 0 | -2 | -29 |
| [src/ts/toolbar/Br.ts](/src/ts/toolbar/Br.ts) | TypeScript | -7 | 0 | -2 | -9 |
| [src/ts/toolbar/CodeTheme.ts](/src/ts/toolbar/CodeTheme.ts) | TypeScript | -31 | 0 | -6 | -37 |
| [src/ts/toolbar/ContentTheme.ts](/src/ts/toolbar/ContentTheme.ts) | TypeScript | -30 | 0 | -6 | -36 |
| [src/ts/toolbar/Counter.ts](/src/ts/toolbar/Counter.ts) | TypeScript | -36 | 0 | -5 | -41 |
| [src/ts/toolbar/Custom.ts](/src/ts/toolbar/Custom.ts) | TypeScript | -16 | 0 | -2 | -18 |
| [src/ts/toolbar/Devtools.ts](/src/ts/toolbar/Devtools.ts) | TypeScript | -26 | 0 | -3 | -29 |
| [src/ts/toolbar/Divider.ts](/src/ts/toolbar/Divider.ts) | TypeScript | -7 | 0 | -2 | -9 |
| [src/ts/toolbar/EditMode.ts](/src/ts/toolbar/EditMode.ts) | TypeScript | -165 | -5 | -24 | -194 |
| [src/ts/toolbar/Emoji.ts](/src/ts/toolbar/Emoji.ts) | TypeScript | -68 | 0 | -7 | -75 |
| [src/ts/toolbar/Export.ts](/src/ts/toolbar/Export.ts) | TypeScript | -39 | 0 | -3 | -42 |
| [src/ts/toolbar/Fullscreen.ts](/src/ts/toolbar/Fullscreen.ts) | TypeScript | -55 | 0 | -7 | -62 |
| [src/ts/toolbar/Headings.ts](/src/ts/toolbar/Headings.ts) | TypeScript | -62 | 0 | -8 | -70 |
| [src/ts/toolbar/Help.ts](/src/ts/toolbar/Help.ts) | TypeScript | -29 | 0 | -2 | -31 |
| [src/ts/toolbar/Indent.ts](/src/ts/toolbar/Indent.ts) | TypeScript | -23 | 0 | -3 | -26 |
| [src/ts/toolbar/Info.ts](/src/ts/toolbar/Info.ts) | TypeScript | -42 | 0 | -2 | -44 |
| [src/ts/toolbar/InsertAfter.ts](/src/ts/toolbar/InsertAfter.ts) | TypeScript | -17 | 0 | -2 | -19 |
| [src/ts/toolbar/InsertBefore.ts](/src/ts/toolbar/InsertBefore.ts) | TypeScript | -17 | 0 | -2 | -19 |
| [src/ts/toolbar/MenuItem.ts](/src/ts/toolbar/MenuItem.ts) | TypeScript | -68 | -1 | -5 | -74 |
| [src/ts/toolbar/Outdent.ts](/src/ts/toolbar/Outdent.ts) | TypeScript | -23 | 0 | -2 | -25 |
| [src/ts/toolbar/Outline.ts](/src/ts/toolbar/Outline.ts) | TypeScript | -20 | 0 | -2 | -22 |
| [src/ts/toolbar/Preview.ts](/src/ts/toolbar/Preview.ts) | TypeScript | -52 | 0 | -4 | -56 |
| [src/ts/toolbar/Record.ts](/src/ts/toolbar/Record.ts) | TypeScript | -54 | -2 | -5 | -61 |
| [src/ts/toolbar/Redo.ts](/src/ts/toolbar/Redo.ts) | TypeScript | -17 | 0 | -2 | -19 |
| [src/ts/toolbar/Undo.ts](/src/ts/toolbar/Undo.ts) | TypeScript | -17 | 0 | -2 | -19 |
| [src/ts/toolbar/Upload.ts](/src/ts/toolbar/Upload.ts) | TypeScript | -39 | 0 | -3 | -42 |
| [src/ts/toolbar/index.ts](/src/ts/toolbar/index.ts) | TypeScript | -166 | 0 | -10 | -176 |
| [src/ts/toolbar/setToolbar.ts](/src/ts/toolbar/setToolbar.ts) | TypeScript | -164 | -6 | -9 | -179 |
| [src/ts/ui/initUI.ts](/src/ts/ui/initUI.ts) | TypeScript | -188 | -5 | -31 | -224 |
| [src/ts/ui/setCodeTheme.ts](/src/ts/ui/setCodeTheme.ts) | TypeScript | -15 | 0 | -2 | -17 |
| [src/ts/ui/setContentTheme.ts](/src/ts/ui/setContentTheme.ts) | TypeScript | -14 | 0 | -2 | -16 |
| [src/ts/ui/setPreviewMode.ts](/src/ts/ui/setPreviewMode.ts) | TypeScript | -26 | 0 | -6 | -32 |
| [src/ts/ui/setTheme.ts](/src/ts/ui/setTheme.ts) | TypeScript | -7 | 0 | -1 | -8 |
| [src/ts/undo/index.ts](/src/ts/undo/index.ts) | TypeScript | -225 | -10 | -24 | -259 |
| [src/ts/upload/getElement.ts](/src/ts/upload/getElement.ts) | TypeScript | -10 | 0 | -1 | -11 |
| [src/ts/upload/index.ts](/src/ts/upload/index.ts) | TypeScript | -230 | -1 | -30 | -261 |
| [src/ts/upload/setHeaders.ts](/src/ts/upload/setHeaders.ts) | TypeScript | -10 | 0 | -1 | -11 |
| [src/ts/util/Options.ts](/src/ts/util/Options.ts) | TypeScript | -423 | -4 | -8 | -435 |
| [src/ts/util/RecordMedia.ts](/src/ts/util/RecordMedia.ts) | TypeScript | -123 | -20 | -28 | -171 |
| [src/ts/util/addScript.ts](/src/ts/util/addScript.ts) | TypeScript | -9 | -37 | 0 | -46 |
| [src/ts/util/addStyle.ts](/src/ts/util/addStyle.ts) | TypeScript | -10 | 0 | -1 | -11 |
| [src/ts/util/code160to32.ts](/src/ts/util/code160to32.ts) | TypeScript | -3 | -1 | -1 | -5 |
| [src/ts/util/compatibility.ts](/src/ts/util/compatibility.ts) | TypeScript | -57 | -5 | -6 | -68 |
| [src/ts/util/editorCommonEvent.ts](/src/ts/util/editorCommonEvent.ts) | TypeScript | -224 | -12 | -19 | -255 |
| [src/ts/util/fixBrowserBehavior.ts](/src/ts/util/fixBrowserBehavior.ts) | TypeScript | -1,326 | -76 | -90 | -1,492 |
| [src/ts/util/getSelectText.ts](/src/ts/util/getSelectText.ts) | TypeScript | -7 | 0 | -2 | -9 |
| [src/ts/util/hasClosest.ts](/src/ts/util/hasClosest.ts) | TypeScript | -164 | -6 | -13 | -183 |
| [src/ts/util/hasClosestByHeadings.ts](/src/ts/util/hasClosestByHeadings.ts) | TypeScript | -25 | -1 | -2 | -28 |
| [src/ts/util/highlightToolbar.ts](/src/ts/util/highlightToolbar.ts) | TypeScript | -9 | 0 | -2 | -11 |
| [src/ts/util/hotKey.ts](/src/ts/util/hotKey.ts) | TypeScript | -44 | -5 | -6 | -55 |
| [src/ts/util/log.ts](/src/ts/util/log.ts) | TypeScript | -5 | -1 | -1 | -7 |
| [src/ts/util/merge.ts](/src/ts/util/merge.ts) | TypeScript | -18 | 0 | -1 | -19 |
| [src/ts/util/processCode.ts](/src/ts/util/processCode.ts) | TypeScript | -82 | -4 | -5 | -91 |
| [src/ts/util/selection.ts](/src/ts/util/selection.ts) | TypeScript | -242 | -13 | -21 | -276 |
| [src/ts/util/toc.ts](/src/ts/util/toc.ts) | TypeScript | -77 | -3 | -4 | -84 |
| [src/ts/wysiwyg/afterRenderEvent.ts](/src/ts/wysiwyg/afterRenderEvent.ts) | TypeScript | 3 | 1 | 0 | 4 |
| [src/ts/wysiwyg/highlightToolbarWYSIWYG.ts](/src/ts/wysiwyg/highlightToolbarWYSIWYG.ts) | TypeScript | 206 | 5 | 1 | 212 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details