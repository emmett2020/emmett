// new一个新的Lute，然后根据options去填充这个lute，然后返回这个lute
// 这斜函数都定义在lute里，
export const setLute = (options: ILuteOptions) => {
    const lute: Lute = Lute.New()
    lute.PutEmojis(options.emojis)
    lute.SetEmojiSite(options.emojiSite)
    lute.SetHeadingAnchor(options.headingAnchor)
    lute.SetInlineMathAllowDigitAfterOpenMarker(options.inlineMathDigit)
    lute.SetAutoSpace(options.autoSpace)
    lute.SetToC(options.toc)
    lute.SetFootnotes(options.footnotes)
    lute.SetFixTermTypo(options.fixTermTypo)
    lute.SetVditorCodeBlockPreview(options.codeBlockPreview)
    lute.SetVditorMathBlockPreview(options.mathBlockPreview)
    lute.SetSanitize(options.sanitize)
    lute.SetChineseParagraphBeginningSpace(options.paragraphBeginningSpace)
    lute.SetRenderListStyle(options.listStyle)
    lute.SetLinkBase(options.linkBase)
    lute.SetLinkPrefix(options.linkPrefix)
    lute.SetMark(options.mark)
    if (options.lazyLoadImage) {
        lute.SetImageLazyLoading(options.lazyLoadImage)
    }
    return lute
}
