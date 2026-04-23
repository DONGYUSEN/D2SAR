import warnings

from diagnostics.rtc_manual import main, run_manual_rtc


if __name__ == "__main__":
    warnings.warn(
        "scripts/rtc_manual.py is a compatibility wrapper; use scripts/diagnostics/rtc_manual.py via this entrypoint instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
