import warnings

from diagnostics.tianyi_rtc_fallback import compute_tianyi_rtc_fallback, main


if __name__ == "__main__":
    warnings.warn(
        "scripts/tianyi_rtc_fallback.py is a compatibility wrapper; use scripts/diagnostics/tianyi_rtc_fallback.py via this entrypoint instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
